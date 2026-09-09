"""三层验证评测脚本（见 docs/superpowers/specs/2026-09-10-repo-audit-design.md §1.1）。

子命令::

    python scripts/evaluate.py calibrate  [--per-gender 300]   # AISHELL-3 → 韵律基准 μ/σ
    python scripts/evaluate.py neutral    [--limit 200]        # AISHELL-3 → CER + 中性语音输出分布
    python scripts/evaluate.py emotion    [--per-cell 6]       # CSEMOTIONS → 象限/方向/消融
    python scripts/evaluate.py report                          # 汇总缓存结果为 Markdown 表格

数据来源与许可（**不随仓库分发任何音频**，运行时按需从 HuggingFace 下载到
``portable_data/eval/``，该目录已 gitignore）：

- AISHELL-3（OpenSLR 93，Apache-2.0）：HF 镜像 ``shenyunhang/AISHELL-3``，
  按需下载 ``test/`` 下的单个 wav 与 ``content.txt``、``spk-info.txt``。
- CSEMOTIONS（AIDC-AI，Apache-2.0；其 NOTICE 声明含 HLTSingapore ESD 衍生内容，
  仅限研究用途）：HF ``AIDC-AI/CSEMOTIONS`` parquet 分片。

实现说明：不依赖 ``datasets`` 库（其 import 会触发 Windows 证书库加载，在部分机器
上因证书损坏抛 SSLError），只用 ``huggingface_hub`` + ``pyarrow`` + ``soundfile``。
每条语音的原始模态分数缓存为 JSON，消融与动态权重开关在缓存上离线重算，无需
重跑模型。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 评测数据从官方 Hub 下载：hf-mirror.com 对这两个数据集返回 308 跳转到 huggingface.co，
# 会使 huggingface_hub 的元数据校验失败。用户仍可通过环境变量 HF_ENDPOINT 覆盖。
import os  # noqa: E402

os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

from src import portable  # noqa: E402  触发便携环境（HF 缓存重定向到 portable_data）

EVAL_ROOT = portable.PORTABLE_DATA_ROOT / "eval"
AISHELL_DIR = EVAL_ROOT / "aishell3"
CSEM_DIR = EVAL_ROOT / "csemotions"
RESULTS_DIR = EVAL_ROOT / "results"
NORMS_PATH = portable.CONFIG_DIR / "prosody_norms.json"

AISHELL_REPO = "shenyunhang/AISHELL-3"
CSEM_REPO = "AIDC-AI/CSEMOTIONS"

SEED = 20260910
_PUNCT_RE = re.compile(r"[，。！？、；：,.!?;:\"'“”‘’（）()《》<>【】\[\]…—\-\s]+")

# CSEMOTIONS 7 类 → 参照象限。neutral / surprise 单独报告，不计入严格象限准确率
# （neutral 无象限归属；surprise 的效价在文献中不稳定，见 research_notes §3.1）。
EMOTION_TO_QUADRANT = {
    "happy": "Q1", "playfulness": "Q1",
    "angry": "Q2", "fearful": "Q2",
    "sad": "Q3",
}
# 方向一致性检验用的 V-A 序（Russell 1980 环形位置，仅比较相对高低）
EXPECTED_AROUSAL_ORDER = ["sad", "neutral", "happy", "angry", "fearful"]   # 低→高
EXPECTED_VALENCE_ORDER = ["angry", "fearful", "sad", "neutral", "happy"]   # 低→高


# ---------------------------------------------------------------------- #
# 工具
# ---------------------------------------------------------------------- #
def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _strip_punct(text: str) -> str:
    return _PUNCT_RE.sub("", text or "")


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_std(xs: Iterable[float]) -> tuple[float, float, int]:
    a = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=np.float64)
    if len(a) == 0:
        return float("nan"), float("nan"), 0
    return float(a.mean()), float(a.std(ddof=1) if len(a) > 1 else 0.0), int(len(a))


def _hf_download(repo: str, filename: str, local_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    local_dir.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo, filename, repo_type="dataset", local_dir=str(local_dir)))


# ---------------------------------------------------------------------- #
# AISHELL-3
# ---------------------------------------------------------------------- #
@dataclass
class AishellUtt:
    spk: str
    gender: str
    wav_rel: str      # test/wav/SSBxxxx/SSBxxxx0001.wav
    text: str         # 纯汉字


def aishell_index() -> list[AishellUtt]:
    """读取 spk-info.txt 与 test/content.txt，返回测试集全部语句索引。"""
    spk_info = _hf_download(AISHELL_REPO, "spk-info.txt", AISHELL_DIR)
    content = _hf_download(AISHELL_REPO, "test/content.txt", AISHELL_DIR)
    gender: dict[str, str] = {}
    for line in spk_info.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            gender[parts[0].strip()] = parts[2].strip().lower()
    utts: list[AishellUtt] = []
    for line in content.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        fname, trans = line.split("\t", 1)
        spk = fname[:7]
        toks = trans.split()
        hanzi = "".join(t for i, t in enumerate(toks) if i % 2 == 0)
        if spk in gender:
            utts.append(AishellUtt(spk, gender[spk], f"test/wav/{spk}/{fname}", hanzi))
    return utts


SPEAKERS_PER_GENDER = 10   # 每性别抽取的说话人数


def aishell_sample(per_gender: int, seed: int = SEED) -> list[AishellUtt]:
    """性别均衡的分层抽样：每性别固定抽 SPEAKERS_PER_GENDER 位说话人，再在说话人内均分名额。

    抽样后只并行下载被选中的文件（snapshot_download 以精确路径为 allow_patterns），
    避免逐文件 HEAD+GET 的高延迟。
    """
    rng = random.Random(seed)
    by_gender: dict[str, dict[str, list[AishellUtt]]] = defaultdict(lambda: defaultdict(list))
    for u in aishell_index():
        by_gender[u.gender][u.spk].append(u)
    out: list[AishellUtt] = []
    for g in ("female", "male"):
        spks = sorted(by_gender[g])
        rng.shuffle(spks)
        spks = spks[:SPEAKERS_PER_GENDER]
        quota = max(1, per_gender // max(1, len(spks)))
        picked: list[AishellUtt] = []
        for spk in spks:
            pool = sorted(by_gender[g][spk], key=lambda u: u.wav_rel)
            rng.shuffle(pool)
            picked.extend(pool[:quota])
        rng.shuffle(picked)
        out.extend(picked[:per_gender])
    _aishell_fetch([u.wav_rel for u in out])
    return out


def _aishell_fetch(paths: list[str]) -> None:
    """并行下载指定文件（仅缺失的）。"""
    from huggingface_hub import snapshot_download
    missing = [rel for rel in paths if not (AISHELL_DIR / rel).exists()]
    if not missing:
        return
    _log(f"snapshot_download {len(missing)} 个缺失文件（8 并发）…")
    snapshot_download(AISHELL_REPO, repo_type="dataset", local_dir=str(AISHELL_DIR),
                      allow_patterns=missing, max_workers=8)


def aishell_wav(u: AishellUtt) -> Path:
    p = AISHELL_DIR / u.wav_rel
    if not p.exists():
        return _hf_download(AISHELL_REPO, u.wav_rel, AISHELL_DIR)
    return p


# ---------------------------------------------------------------------- #
# CSEMOTIONS
# ---------------------------------------------------------------------- #
def csem_shards() -> list[str]:
    from huggingface_hub import list_repo_files
    return sorted(f for f in list_repo_files(CSEM_REPO, repo_type="dataset")
                  if f.startswith("data/") and f.endswith(".parquet"))


def csem_iter(shards: list[str]):
    """逐分片读取 parquet，yield (meta dict, wav bytes)。"""
    import pyarrow.parquet as pq
    for shard in shards:
        path = _hf_download(CSEM_REPO, shard, CSEM_DIR)
        table = pq.read_table(path)
        cols = table.column_names
        for row in table.to_pylist():
            audio = row.get("audio") or {}
            yield {
                "emotion": str(row.get("emotion", "")).strip().lower(),
                "speaker": str(row.get("speaker", row.get("speaker_id", ""))),
                "text": row.get("text", row.get("transcript", "")) or "",
                "path": audio.get("path", ""),
                "_cols": cols,
            }, audio.get("bytes")


def csem_sample(per_cell: int, seed: int = SEED) -> list[dict[str, Any]]:
    """每（情绪 × 说话人）单元抽 per_cell 句，写成本地 wav，返回元数据列表。"""
    rng = random.Random(seed)
    wav_dir = CSEM_DIR / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    index_path = CSEM_DIR / f"sample_pc{per_cell}_seed{seed}.json"
    if index_path.exists():
        return _json_load(index_path)

    _log("扫描 CSEMOTIONS 分片（首次需下载约 3 GB parquet）…")
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: Counter = Counter()
    n_total = 0
    for meta, wav_bytes in csem_iter(csem_shards()):
        n_total += 1
        key = (meta["emotion"], meta["speaker"])
        seen[key] += 1
        # 每单元独立的蓄水池抽样（Algorithm R）：保留 per_cell 条，只落盘被选中的 wav
        bucket = cells[key]
        if len(bucket) < per_cell:
            slot = len(bucket)
        else:
            j = rng.randrange(seen[key])
            if j >= per_cell:
                continue
            slot = j
        fname = f"{meta['emotion']}_{meta['speaker']}_{n_total:05d}.wav"
        out = wav_dir / fname
        out.write_bytes(wav_bytes)
        rec = {"emotion": meta["emotion"], "speaker": meta["speaker"],
               "text": meta["text"], "wav": str(out.relative_to(CSEM_DIR))}
        if slot < len(bucket):
            old = bucket[slot]
            try:
                (CSEM_DIR / old["wav"]).unlink()
            except OSError:
                pass
            bucket[slot] = rec
        else:
            bucket.append(rec)
    sample = [r for b in cells.values() for r in b]
    _log(f"CSEMOTIONS 共 {n_total} 条，抽样 {len(sample)} 条，"
         f"{len({r['emotion'] for r in sample})} 类情绪 × {len({r['speaker'] for r in sample})} 位说话人")
    _json_dump(sample, index_path)
    return sample


# ---------------------------------------------------------------------- #
# 管线封装
# ---------------------------------------------------------------------- #
class _Runner:
    def __init__(self) -> None:
        from src.models.model_manager import ModelManager
        from src.pipeline import AnalysisPipeline
        _log("加载 4 个模型…")
        self.manager = ModelManager()
        self.manager.load_all(progress_cb=lambda s, p: _log(f"  [{p:3d}%] {s}"))
        self.pipeline = AnalysisPipeline(self.manager)
        self.config = self.pipeline.config

    def analyze(self, wav: Path) -> dict[str, Any]:
        r = self.pipeline.analyze(str(wav))
        # 只保留离线重算所需字段
        return {
            "negative": r["negative"], "valence": r["valence"], "arousal": r["arousal"],
            "dominant_quadrant": r["dominant_quadrant"], "memberships": r["memberships"],
            "modal_scores": r["modal_scores"], "weights": r["weights"],
            "asr_text": r["asr_text"], "asr_confidence": r["asr_confidence"],
            "snr_db": r["audio_quality"]["snr_db"], "duration": r["duration"],
            "paralang_events": r["paralang_events"],
            "degraded_modalities": r.get("degraded_modalities", []),
            "prosody": r["modal_details"].get("prosody", {}),
        }


def _refuse(config: dict, rec: dict, *, drop: set[str] = frozenset(),
            only: str | None = None, dynamic: bool = True) -> dict[str, Any]:
    """在缓存的模态分数上离线重算融合（消融 / 动态权重开关）。"""
    from src.fusion.weighted_fusion import MODALITIES, WeightedFusion
    fus = WeightedFusion(config)
    scores = {}
    for m in MODALITIES:
        s = rec["modal_scores"][m]
        if m in drop or (only is not None and m != only):
            scores[m] = (0.5, 0.5)
        else:
            scores[m] = (s["negative"], s["arousal"])
    if dynamic:
        return fus.fuse(scores, audio_quality={"snr_db": rec["snr_db"]},
                        asr_confidence=rec["asr_confidence"],
                        paralang_events=rec["paralang_events"])
    return fus.fuse(scores)  # 默认参数 = 静态权重（SNR 15 dB、ASR 0.8、无事件）


# ---------------------------------------------------------------------- #
# calibrate
# ---------------------------------------------------------------------- #
def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False



FEATURES = ("mean_f0", "std_f0", "f0_range", "speech_rate", "pause_ratio",
            "hnr", "jitter_local", "shimmer_local", "f0_slope")


def cmd_calibrate(args: argparse.Namespace) -> int:
    from src.audio import loader
    from src.features import prosody

    from src.audio import vad
    from src.models.asr_model import ASRModel

    sample = aishell_sample(args.per_gender)
    _log(f"AISHELL-3 分层抽样 {len(sample)} 句（每性别 {args.per_gender}）")
    # 语速定义必须与管线一致：管线用 ASR 字级时间戳覆盖的语音时长（vad.syllable_rate）。
    # 若改用「字数 / 含静音总时长」，基准会系统性偏低，导致所有语音看起来「偏快」→ 唤醒度虚高。
    _log("加载 Paraformer（仅用于取字级时间戳）…")
    asr = ASRModel(device="cuda" if _cuda_available() else "cpu")
    rows: list[dict[str, Any]] = []
    for i, u in enumerate(sample, 1):
        wav = aishell_wav(u)
        y, sr = loader.load_audio(wav, target_sr=16000)
        dur = len(y) / sr
        asr_res = asr.transcribe(str(wav))
        # 字数用标注转写（真值），时长用 ASR 时间戳覆盖的语音段（与管线同一函数）
        rate = vad.syllable_rate({"text": u.text, "timestamp": asr_res.get("timestamp", [])})
        if rate is None:
            rate = len(u.text) / dur if dur > 0 else None  # 无时间戳时退化
        feat = prosody.extract(y, sr, syllable_rate=rate)
        row = {"spk": u.spk, "gender": u.gender, "duration": dur, "n_chars": len(u.text),
               "asr_text": asr_res.get("text", "")}
        row.update({k: getattr(feat, k) for k in FEATURES})
        rows.append(row)
        if i % 50 == 0:
            _log(f"  {i}/{len(sample)}")
    _json_dump(rows, RESULTS_DIR / "calibrate_rows.json")

    def stats(subset: list[dict[str, Any]]) -> dict[str, Any]:
        out = {}
        for k in FEATURES:
            mu, sd, n = _mean_std(r[k] for r in subset)
            out[k] = {"mu": round(mu, 4), "sigma": round(sd, 4), "n": n}
        return out

    norms = {
        "_meta": {
            "source": "AISHELL-3 test set (OpenSLR 93, Apache-2.0), HF mirror shenyunhang/AISHELL-3",
            "sampling": f"stratified by gender, {args.per_gender} per gender, "
                        f"quota per speaker, seed={SEED}",
            "n_total": len(rows),
            "n_female": sum(r["gender"] == "female" for r in rows),
            "n_male": sum(r["gender"] == "male" for r in rows),
            "speech_rate_definition": "transcript characters / duration covered by Paraformer char-level "
                                      "timestamps (same as pipeline vad.syllable_rate)",
            "extractor": "src.features.prosody.extract (parselmouth, 16 kHz)",
            "generated": date.today().isoformat(),
            "script": "scripts/evaluate.py calibrate",
        },
        "mixed": stats(rows),
        "female": stats([r for r in rows if r["gender"] == "female"]),
        "male": stats([r for r in rows if r["gender"] == "male"]),
    }
    _json_dump(norms, NORMS_PATH)
    _log(f"已写入 {NORMS_PATH}")
    for k in FEATURES:
        m = norms["mixed"][k]
        _log(f"  {k:14s} mu={m['mu']:9.4f} sigma={m['sigma']:8.4f}  "
             f"(F mu={norms['female'][k]['mu']:.3f} / M mu={norms['male'][k]['mu']:.3f})")
    return 0


# ---------------------------------------------------------------------- #
# neutral
# ---------------------------------------------------------------------- #
def cmd_neutral(args: argparse.Namespace) -> int:
    import jiwer
    sample = aishell_sample(args.limit // 2)
    _log(f"AISHELL-3 中性语音全管线评测 {len(sample)} 句")
    runner = _Runner()
    cache = RESULTS_DIR / "neutral_rows.json"
    rows: list[dict[str, Any]] = _json_load(cache) if cache.exists() else []
    done = {r["wav_rel"] for r in rows}
    for i, u in enumerate(sample, 1):
        if u.wav_rel in done:
            continue
        r = runner.analyze(aishell_wav(u))
        r.update({"wav_rel": u.wav_rel, "spk": u.spk, "gender": u.gender, "ref_text": u.text})
        rows.append(r)
        if i % 10 == 0:
            _json_dump(rows, cache)
            _log(f"  {i}/{len(sample)}")
    _json_dump(rows, cache)

    refs = [_strip_punct(r["ref_text"]) for r in rows]
    hyps = [_strip_punct(r["asr_text"]) for r in rows]
    # 字级 CER：把每个字当作一个 token
    cer = jiwer.cer([" ".join(x) for x in refs], [" ".join(x) for x in hyps]) if any(refs) else float("nan")
    summary = {
        "n": len(rows),
        "cer": cer,
        "asr_confidence_proxy": _mean_std(r["asr_confidence"] for r in rows),
        "snr_db": _mean_std(r["snr_db"] for r in rows),
        "negative": _mean_std(r["negative"] for r in rows),
        "valence": _mean_std(r["valence"] for r in rows),
        "arousal": _mean_std(r["arousal"] for r in rows),
        "quadrant_hist": dict(Counter(r["dominant_quadrant"] for r in rows)),
        "modal_means": {m: {ax: _mean_std(r["modal_scores"][m][ax] for r in rows)[0]
                            for ax in ("negative", "arousal")}
                        for m in rows[0]["modal_scores"]} if rows else {},
        "degraded_counts": dict(Counter(m for r in rows for m in r["degraded_modalities"])),
    }
    _json_dump(summary, RESULTS_DIR / "neutral_summary.json")
    _log(f"CER={cer:.3f}  negative μ={summary['negative'][0]:.3f}  "
         f"arousal μ={summary['arousal'][0]:.3f}  象限分布={summary['quadrant_hist']}")
    return 0


# ---------------------------------------------------------------------- #
# emotion
# ---------------------------------------------------------------------- #
def _quadrant_metrics(rows: list[dict[str, Any]], key_q: str = "dominant_quadrant") -> dict[str, Any]:
    labeled = [r for r in rows if r["emotion"] in EMOTION_TO_QUADRANT]
    conf: dict[str, Counter] = defaultdict(Counter)
    correct = 0
    for r in labeled:
        ref = EMOTION_TO_QUADRANT[r["emotion"]]
        conf[r["emotion"]][r[key_q]] += 1
        correct += r[key_q] == ref
    return {
        "n": len(labeled),
        "accuracy": correct / len(labeled) if labeled else float("nan"),
        "confusion": {e: dict(c) for e, c in conf.items()},
    }


def _direction_metrics(rows: list[dict[str, Any]], key_v="valence", key_a="arousal") -> dict[str, Any]:
    from scipy.stats import spearmanr
    by_emo: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_emo[r["emotion"]].append(r)
    means = {e: {"valence": _mean_std(x[key_v] for x in rs),
                 "arousal": _mean_std(x[key_a] for x in rs),
                 "negative": _mean_std(x["negative"] for x in rs)}
             for e, rs in by_emo.items()}
    # Spearman：把每条语音的类别参照序作为等级
    def rho(order: list[str], key: str) -> float:
        rank = {e: i for i, e in enumerate(order)}
        xs = [rank[r["emotion"]] for r in rows if r["emotion"] in rank]
        ys = [r[key] for r in rows if r["emotion"] in rank]
        return float(spearmanr(xs, ys).correlation) if len(set(xs)) > 1 else float("nan")
    # 成对方向检验
    def pair_ok(lo: str, hi: str, key: str) -> bool | None:
        if lo not in means or hi not in means:
            return None
        return means[lo][key][0] < means[hi][key][0]
    pairs = {
        "arousal: sad < angry": pair_ok("sad", "angry", "arousal"),
        "arousal: sad < fearful": pair_ok("sad", "fearful", "arousal"),
        "arousal: neutral < happy": pair_ok("neutral", "happy", "arousal"),
        "arousal: neutral < angry": pair_ok("neutral", "angry", "arousal"),
        "valence: angry < happy": pair_ok("angry", "happy", "valence"),
        "valence: sad < happy": pair_ok("sad", "happy", "valence"),
        "valence: fearful < neutral": pair_ok("fearful", "neutral", "valence"),
        "valence: sad < neutral": pair_ok("sad", "neutral", "valence"),
    }
    return {
        "per_emotion": means,
        "spearman_arousal": rho(EXPECTED_AROUSAL_ORDER, key_a),
        "spearman_valence": rho(EXPECTED_VALENCE_ORDER, key_v),
        "pairwise": pairs,
        "pairwise_pass": sum(1 for v in pairs.values() if v) ,
        "pairwise_total": sum(1 for v in pairs.values() if v is not None),
    }


def cmd_emotion(args: argparse.Namespace) -> int:
    sample = csem_sample(args.per_cell)
    _log(f"CSEMOTIONS 全管线评测 {len(sample)} 句")
    runner = _Runner()
    cache = RESULTS_DIR / "emotion_rows.json"
    rows: list[dict[str, Any]] = _json_load(cache) if cache.exists() else []
    done = {r["wav"] for r in rows}
    for i, rec in enumerate(sample, 1):
        if rec["wav"] in done:
            continue
        r = runner.analyze(CSEM_DIR / rec["wav"])
        r.update({"wav": rec["wav"], "emotion": rec["emotion"], "speaker": rec["speaker"],
                  "ref_text": rec["text"]})
        rows.append(r)
        if i % 10 == 0:
            _json_dump(rows, cache)
            _log(f"  {i}/{len(sample)}")
    _json_dump(rows, cache)
    return _emotion_summarize(runner.config, rows)


def _emotion_summarize(config: dict, rows: list[dict[str, Any]]) -> int:
    import jiwer

    from src.fusion.weighted_fusion import MODALITIES
    summary: dict[str, Any] = {"n": len(rows),
                               "emotions": dict(Counter(r["emotion"] for r in rows)),
                               "speakers": len({r["speaker"] for r in rows})}
    # 全模态 + 动态权重（= 管线默认）
    summary["full"] = {"quadrant": _quadrant_metrics(rows), "direction": _direction_metrics(rows)}
    # 动态权重关闭
    static_rows = []
    for r in rows:
        f = _refuse(config, r, dynamic=False)
        static_rows.append({**r, "valence": f["valence"], "arousal": f["arousal"],
                            "negative": f["negative"], "dominant_quadrant": f["dominant_quadrant"]})
    summary["static_weights"] = {"quadrant": _quadrant_metrics(static_rows),
                                 "direction": _direction_metrics(static_rows)}
    # 消融：逐一去除 / 仅保留
    summary["ablation_drop"] = {}
    summary["ablation_only"] = {}
    for m in MODALITIES:
        for mode, kw in (("ablation_drop", {"drop": {m}}), ("ablation_only", {"only": m})):
            rr = []
            for r in rows:
                f = _refuse(config, r, **kw)
                rr.append({**r, "valence": f["valence"], "arousal": f["arousal"],
                           "negative": f["negative"], "dominant_quadrant": f["dominant_quadrant"]})
            q = _quadrant_metrics(rr)
            d = _direction_metrics(rr)
            summary[mode][m] = {"accuracy": q["accuracy"], "spearman_arousal": d["spearman_arousal"],
                                "spearman_valence": d["spearman_valence"],
                                "pairwise_pass": d["pairwise_pass"]}
    # ASR 质量（表演型语音）
    refs = [_strip_punct(r["ref_text"]) for r in rows]
    hyps = [_strip_punct(r["asr_text"]) for r in rows]
    summary["cer"] = jiwer.cer([" ".join(x) for x in refs], [" ".join(x) for x in hyps]) if any(refs) else float("nan")
    summary["snr_db"] = _mean_std(r["snr_db"] for r in rows)
    summary["asr_confidence_proxy"] = _mean_std(r["asr_confidence"] for r in rows)
    summary["degraded_counts"] = dict(Counter(m for r in rows for m in r["degraded_modalities"]))
    _json_dump(summary, RESULTS_DIR / "emotion_summary.json")
    q = summary["full"]["quadrant"]
    d = summary["full"]["direction"]
    _log(f"象限准确率={q['accuracy']:.3f} (n={q['n']})  Spearman A={d['spearman_arousal']:.3f} "
         f"V={d['spearman_valence']:.3f}  方向检验 {d['pairwise_pass']}/{d['pairwise_total']}")
    return 0


# ---------------------------------------------------------------------- #
# report
# ---------------------------------------------------------------------- #
def _fmt(x: Any, nd: int = 3) -> str:
    if isinstance(x, (int, np.integer)):
        return str(x)
    if isinstance(x, float) and np.isfinite(x):
        return f"{x:.{nd}f}"
    return "—"


def cmd_report(args: argparse.Namespace) -> int:
    out: list[str] = []
    if NORMS_PATH.exists():
        norms = _json_load(NORMS_PATH)
        meta = norms["_meta"]
        out.append("### 第 1 层：韵律基准实测（AISHELL-3）\n")
        out.append(f"样本：{meta['n_total']} 句（女 {meta['n_female']} / 男 {meta['n_male']}），"
                   f"{meta['sampling']}，生成日期 {meta['generated']}。\n")
        out.append("| 特征 | 混合 μ | 混合 σ | 女 μ | 男 μ | 旧启发式 μ/σ |")
        out.append("|------|------:|------:|-----:|-----:|:------------:|")
        from src.fusion.normalizer import LEGACY_PROSODY_STATS
        for k in FEATURES:
            m = norms["mixed"][k]
            f = norms["female"][k]
            mm = norms["male"][k]
            old = LEGACY_PROSODY_STATS.get(k)
            out.append(f"| {k} | {m['mu']:.3f} | {m['sigma']:.3f} | {f['mu']:.3f} | {mm['mu']:.3f} | "
                       f"{old[0]:.3f} / {old[1]:.3f} |" if old else
                       f"| {k} | {m['mu']:.3f} | {m['sigma']:.3f} | {f['mu']:.3f} | {mm['mu']:.3f} | — |")
        out.append("")
    p = RESULTS_DIR / "neutral_summary.json"
    if p.exists():
        s = _json_load(p)
        out.append("### 第 1 层副产物：中性语音全管线（AISHELL-3）\n")
        out.append("| 指标 | 值 |\n|------|---:|")
        out.append(f"| 样本数 | {s['n']} |")
        out.append(f"| Paraformer 字错率 CER | {_fmt(s['cer'])} |")
        out.append(f"| ASR 置信度代理 μ (σ) | {_fmt(s['asr_confidence_proxy'][0])} ({_fmt(s['asr_confidence_proxy'][1])}) |")
        out.append(f"| SNR μ dB | {_fmt(s['snr_db'][0], 1)} |")
        for k in ("negative", "valence", "arousal"):
            out.append(f"| {k} μ (σ) | {_fmt(s[k][0])} ({_fmt(s[k][1])}) |")
        out.append(f"| 象限分布 | {s['quadrant_hist']} |")
        if s.get("degraded_counts"):
            out.append(f"| 降级模态计数 | {s['degraded_counts']} |")
        out.append("\n各模态在中性语音上的均值：\n")
        out.append("| 模态 | negative μ | arousal μ |\n|------|---:|---:|")
        for m, v in s["modal_means"].items():
            out.append(f"| {m} | {_fmt(v['negative'])} | {_fmt(v['arousal'])} |")
        out.append("")
    p = RESULTS_DIR / "emotion_summary.json"
    if p.exists():
        s = _json_load(p)
        out.append("### 第 2 层：情绪判别（CSEMOTIONS）\n")
        out.append(f"样本：{s['n']} 句，{s['speakers']} 位说话人，类别计数 {s['emotions']}；"
                   f"CER={_fmt(s['cer'])}，SNR μ={_fmt(s['snr_db'][0],1)} dB，"
                   f"ASR 置信度代理 μ={_fmt(s['asr_confidence_proxy'][0])}。\n")
        d = s["full"]["direction"]
        out.append("**各情绪的 V-A 均值（全模态 + 动态权重）**\n")
        out.append("| 情绪 | n | valence μ (σ) | arousal μ (σ) |\n|------|--:|---:|---:|")
        for e in ("neutral", "happy", "playfulness", "surprise", "angry", "fearful", "sad"):
            if e in d["per_emotion"]:
                v = d["per_emotion"][e]
                out.append(f"| {e} | {v['valence'][2]} | {_fmt(v['valence'][0])} ({_fmt(v['valence'][1])}) | "
                           f"{_fmt(v['arousal'][0])} ({_fmt(v['arousal'][1])}) |")
        out.append("")
        out.append("**方向一致性检验**（成对均值比较，期望 lo < hi）\n")
        out.append("| 检验 | 结果 |\n|------|:---:|")
        for k, v in d["pairwise"].items():
            out.append(f"| {k} | {'✅' if v else ('❌' if v is not None else '—')} |")
        out.append(f"| Spearman ρ（arousal vs 参照序） | {_fmt(d['spearman_arousal'])} |")
        out.append(f"| Spearman ρ（valence vs 参照序） | {_fmt(d['spearman_valence'])} |")
        out.append("")
        q = s["full"]["quadrant"]
        out.append(f"**象限混淆矩阵**（仅 happy/playfulness→Q1、angry/fearful→Q2、sad→Q3 计入，n={q['n']}，"
                   f"准确率 **{_fmt(q['accuracy'])}**；随机基线 0.25，多数类基线见下）\n")
        out.append("| 情绪（参照） | Q1 | Q2 | Q3 | Q4 |\n|------|--:|--:|--:|--:|")
        for e, c in q["confusion"].items():
            out.append(f"| {e}（{EMOTION_TO_QUADRANT[e]}） | " + " | ".join(str(c.get(x, 0)) for x in ("Q1", "Q2", "Q3", "Q4")) + " |")
        out.append("")
        sq = s["static_weights"]["quadrant"]
        sd = s["static_weights"]["direction"]
        out.append("**动态权重开/关**\n")
        out.append("| 设置 | 象限准确率 | ρ arousal | ρ valence | 方向检验 |\n|------|---:|---:|---:|:---:|")
        out.append(f"| 动态权重（默认） | {_fmt(q['accuracy'])} | {_fmt(d['spearman_arousal'])} | {_fmt(d['spearman_valence'])} | {d['pairwise_pass']}/{d['pairwise_total']} |")
        out.append(f"| 静态权重 | {_fmt(sq['accuracy'])} | {_fmt(sd['spearman_arousal'])} | {_fmt(sd['spearman_valence'])} | {sd['pairwise_pass']}/{sd['pairwise_total']} |")
        out.append("")
        out.append("**模态消融**（去除单一模态 / 仅保留单一模态，其余置中性分 0.5）\n")
        out.append("| 模态 | 去除后准确率 | 去除后 ρA | 去除后 ρV | 仅保留准确率 | 仅保留 ρA | 仅保留 ρV |\n|------|---:|---:|---:|---:|---:|---:|")
        for m in s["ablation_drop"]:
            a = s["ablation_drop"][m]
            b = s["ablation_only"][m]
            out.append(f"| {m} | {_fmt(a['accuracy'])} | {_fmt(a['spearman_arousal'])} | {_fmt(a['spearman_valence'])} | "
                       f"{_fmt(b['accuracy'])} | {_fmt(b['spearman_arousal'])} | {_fmt(b['spearman_valence'])} |")
        if s.get("degraded_counts"):
            out.append(f"\n降级模态计数：{s['degraded_counts']}")
        out.append("")
    text = "\n".join(out)
    out_path = RESULTS_DIR / "report_tables.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(text)
    _log(f"表格已写入 {out_path}")
    return 0


# ---------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="三层验证评测")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("calibrate", help="AISHELL-3 韵律基准实测 → config/prosody_norms.json")
    p.add_argument("--per-gender", type=int, default=300)
    p = sub.add_parser("neutral", help="AISHELL-3 全管线：CER + 中性语音输出分布")
    p.add_argument("--limit", type=int, default=200)
    p = sub.add_parser("emotion", help="CSEMOTIONS 情绪判别验证")
    p.add_argument("--per-cell", type=int, default=6, help="每（情绪×说话人）抽样句数")
    p = sub.add_parser("resummarize", help="仅在缓存上重算 emotion 汇总（不跑模型）")
    sub.add_parser("report", help="汇总为 Markdown 表格")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.cmd == "calibrate":
        return cmd_calibrate(args)
    if args.cmd == "neutral":
        return cmd_neutral(args)
    if args.cmd == "emotion":
        return cmd_emotion(args)
    if args.cmd == "resummarize":
        from src.config_loader import load_settings
        return _emotion_summarize(load_settings(), _json_load(RESULTS_DIR / "emotion_rows.json"))
    return cmd_report(args)


if __name__ == "__main__":
    sys.exit(main())
