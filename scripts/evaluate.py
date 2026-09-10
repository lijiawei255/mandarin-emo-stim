"""三层验证评测脚本（见 docs/superpowers/specs/2026-09-10-repo-audit-design.md §1.1）。

子命令::

    python scripts/evaluate.py calibrate  [--per-gender 300]   # AISHELL-3 → 韵律基准 μ/σ
    python scripts/evaluate.py neutral    [--limit 200]        # AISHELL-3 → CER + 中性语音输出分布
    python scripts/evaluate.py emotion    [--per-cell 6]       # CSEMOTIONS → 象限/方向/消融
    python scripts/evaluate.py report                          # 汇总缓存结果为 Markdown 表格
    python scripts/evaluate.py sequence                        # v0.5：会话内平滑/滞回的离线序列模拟

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

v0.2 起：
- ``neutral`` 额外把各模态在中性语音上的**原始**均值写成
  ``config/modality_calibration.json``（offset = 0.5 − 均值），供融合层做基线归一化；
- ``emotion`` 按说话人划分 **dev / test**（说话人 id 排序后前半 dev、后半 test），
  融合权重的网格搜索只在 dev 上进行，test 只报告一次，避免在同一批数据上调参与报告。

v0.3 起：
- **性别均衡的 5 折说话人交叉验证**（每折 1 女 + 1 男）替代单次 dev/test 作为正式数字：
  对每折在训练折上选权重 / 拟合岭回归，在留出折上评估，报告均值 ± 标准差；
- **说话人级中性基线**：用每位说话人自己的 neutral 句计算个人偏移（模拟用户级基线校准），
  在其非中性句上评估；
- **可学习融合**：岭回归（src/fusion/learned_fusion.py）的 CV 表现与手工权重对照，
  并用全部数据拟合写出 ``config/learned_fusion.json``（默认不启用）。

v0.4 起：
- **不确定性校准**：在 CV 的留出折预测上，按模态分歧度（uncertainty）三等分，统计每档的
  象限准确率，写出 ``config/uncertainty_thresholds.json`` 供 GUI 显示「可信度」等级；
- **ASR 后验置信度**：报告 Paraformer token 后验均值的分布及其与逐句 CER 的 Spearman 相关，
  作为 ``asr_confidence_threshold`` 取值依据。
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
CALIB_PATH = portable.CONFIG_DIR / "modality_calibration.json"
LEARNED_PATH = portable.CONFIG_DIR / "learned_fusion.json"
UNC_PATH = portable.CONFIG_DIR / "uncertainty_thresholds.json"
OFFSET_CLIP = 0.3
CV_FOLDS = 5
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
            "modal_scores": r["modal_scores"],
            "modal_scores_raw": r.get("modal_scores_raw", r["modal_scores"]),
            "weights": r["weights"],
            "asr_text": r["asr_text"], "asr_confidence": r["asr_confidence"],
            "snr_db": r["audio_quality"]["snr_db"], "duration": r["duration"],
            "paralang_events": r["paralang_events"],
            "degraded_modalities": r.get("degraded_modalities", []),
            "prosody": r["modal_details"].get("prosody", {}),
            "asr_confidence_source": r.get("asr_confidence_source", "proxy"),
        }


def _refuse(config: dict, rec: dict, *, drop: set[str] = frozenset(),
            only: str | None = None, dynamic: bool = True,
            weights: dict[str, dict[str, float]] | None = None,
            offsets: dict[str, tuple[float, float]] | None = None,
            learned=None) -> dict[str, Any]:
    """在缓存的**原始**模态分数上离线重算融合（消融 / 动态权重开关 / 权重覆盖 /
    偏移覆盖 / 可学习融合）。

    中性校准偏移由 WeightedFusion 按 config 自行施加，因此这里必须喂原始分。
    ``offsets`` 给定时**替换**融合器的偏移（用于说话人级基线）；``learned`` 给定时
    用该 LearnedFusion 计算 negative/arousal。
    """
    from src.fusion.weighted_fusion import MODALITIES, WeightedFusion
    if weights is not None:
        config = {**config, "fusion_weights": weights}
    fus = WeightedFusion(config)
    if offsets is not None:
        fus.offsets = dict(offsets)
        fus.calibration_source = "speaker"
    if learned is not None:
        fus.learned = learned
        fus.fusion_mode = "learned"
    raw = rec.get("modal_scores_raw", rec["modal_scores"])
    scores = {}
    for m in MODALITIES:
        s = raw[m]
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
    from src.audio import loader, vad
    from src.features import prosody
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

    # 先由原始分算出**本次**的校准偏移并写盘，再用它离线重算融合输出，
    # 避免管线运行时加载的旧偏移污染「中性语音输出分布」（v0.3 修正）
    raw_means = {m: {ax: _mean_std(r.get("modal_scores_raw", r["modal_scores"])[m][ax] for r in rows)[0]
                     for ax in ("negative", "arousal")}
                 for m in rows[0]["modal_scores"]} if rows else {}
    offsets = {m: {ax: round(max(-OFFSET_CLIP, min(OFFSET_CLIP, 0.5 - v[ax])), 4)
                   for ax in ("negative", "arousal")} for m, v in raw_means.items()}
    if not args.no_write_calibration:
        _json_dump({
            "_meta": {
                "source": "AISHELL-3 test subset (emotion-neutral read speech), raw modality means",
                "definition": f"offset = 0.5 - mean_raw, clipped to ±{OFFSET_CLIP:.1f}; applied additively in WeightedFusion",
                "n": len(rows), "generated": date.today().isoformat(),
                "script": "scripts/evaluate.py neutral",
            },
            "modal_means_raw": raw_means,
            "offsets": offsets,
        }, CALIB_PATH)
        _log(f"中性校准偏移已写入 {CALIB_PATH}: {offsets}")
    from src.config_loader import load_settings
    rows = _apply(load_settings(), rows)   # 用当前（刚写入的）偏移重算 negative/arousal/象限
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
    summary["modal_means_raw"] = raw_means
    summary["offsets"] = offsets
    summary["asr_confidence"] = _asr_confidence_analysis(rows)
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


def _split_dev_test(rows: list[dict[str, Any]]) -> tuple[list[dict], list[dict], list[str], list[str]]:
    spk = sorted({r["speaker"] for r in rows})
    dev_spk, test_spk = spk[: len(spk) // 2], spk[len(spk) // 2:]
    return ([r for r in rows if r["speaker"] in dev_spk],
            [r for r in rows if r["speaker"] in test_spk], dev_spk, test_spk)


def _apply(config: dict, rows: list[dict[str, Any]], **kw) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        f = _refuse(config, r, **kw)
        out.append({**r, "valence": f["valence"], "arousal": f["arousal"],
                    "negative": f["negative"], "dominant_quadrant": f["dominant_quadrant"],
                    "modal_scores": f["modal_scores"], "uncertainty": f.get("uncertainty", {})})
    return out


def _score_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    q = _quadrant_metrics(rows)
    d = _direction_metrics(rows)
    return {"quadrant": q, "direction": d,
            "composite": float(np.nanmean([q["accuracy"], d["spearman_arousal"], d["spearman_valence"]]))}


def _weight_grid(base: dict[str, dict[str, float]]) -> list[tuple[str, dict[str, dict[str, float]]]]:
    """小而透明的网格：text_llm 的 negative 权重 ∈ {0.30, 0.20, 0.10, 0.05}，
    释放的权重按其余模态原比例分配；arousal 权重不动。"""
    grid = []
    for w in (0.30, 0.20, 0.10, 0.05):
        neg = dict(base["negative"])
        freed = neg["text_llm"] - w
        neg["text_llm"] = w
        others = {m: v for m, v in neg.items() if m != "text_llm"}
        tot = sum(others.values())
        for m in others:
            neg[m] = round(others[m] + freed * others[m] / tot, 4)
        grid.append((f"text_llm.neg={w:.2f}", {"negative": neg, "arousal": dict(base["arousal"])}))
    return grid


def _asr_confidence_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """后验置信度分布 + 与逐句 CER 的 Spearman（负相关说明置信度有信息）。"""
    import jiwer
    from scipy.stats import spearmanr
    conf, cer = [], []
    for r in rows:
        ref, hyp = _strip_punct(r.get("ref_text", "")), _strip_punct(r.get("asr_text", ""))
        if not ref:
            continue
        conf.append(float(r["asr_confidence"]))
        cer.append(jiwer.cer(" ".join(ref), " ".join(hyp)) if hyp else 1.0)
    if len(conf) < 10:
        return {}
    conf_a, cer_a = np.array(conf), np.array(cer)
    rho = spearmanr(conf_a, cer_a).correlation
    return {"n": len(conf), "source": Counter(r.get("asr_confidence_source", "proxy") for r in rows).most_common(1)[0][0],
            "mean": float(conf_a.mean()), "sd": float(conf_a.std(ddof=1)),
            "p5": float(np.percentile(conf_a, 5)), "p25": float(np.percentile(conf_a, 25)),
            "median": float(np.median(conf_a)), "min": float(conf_a.min()),
            "spearman_conf_vs_cer": float(rho) if np.isfinite(rho) else None,
            "cer_when_conf_below_p10": float(cer_a[conf_a <= np.percentile(conf_a, 10)].mean()),
            "cer_when_conf_above_p10": float(cer_a[conf_a > np.percentile(conf_a, 10)].mean())}


def _cv_folds(rows: list[dict[str, Any]], k: int = CV_FOLDS) -> list[list[str]]:
    """性别均衡的说话人折：按性别分组后轮转分配（CSEMOTIONS id 形如 female001 / male003）。"""
    spk = sorted({r["speaker"] for r in rows})
    groups: dict[str, list[str]] = defaultdict(list)
    for s in spk:
        groups["f" if s.lower().startswith(("f", "female")) else "m"].append(s)
    folds: list[list[str]] = [[] for _ in range(k)]
    for g in groups.values():
        for i, s in enumerate(g):
            folds[i % k].append(s)
    return [f for f in folds if f]


def _speaker_offsets(rows: list[dict[str, Any]], speaker: str) -> dict[str, tuple[float, float]]:
    """说话人级中性基线：用其 neutral 句的原始均值，offset = 0.5 − 均值（排除 paralang）。"""
    from src.fusion.personal_calibration import compute_offsets
    raws = [r.get("modal_scores_raw", r["modal_scores"]) for r in rows
            if r["speaker"] == speaker and r["emotion"] == "neutral"]
    return {m: (v["negative"], v["arousal"]) for m, v in compute_offsets(raws).items()}


def _metrics_of(rows: list[dict[str, Any]]) -> dict[str, float]:
    q = _quadrant_metrics(rows)
    d = _direction_metrics(rows)
    return {"accuracy": q["accuracy"], "rhoA": d["spearman_arousal"],
            "rhoV": d["spearman_valence"], "pairwise_pass": d["pairwise_pass"], "n": q["n"]}


def _agg(per_fold: list[dict[str, float]]) -> dict[str, Any]:
    out: dict[str, Any] = {"folds": per_fold}
    for k in ("accuracy", "rhoA", "rhoV", "pairwise_pass"):
        vals = np.asarray([f[k] for f in per_fold], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        out[k] = {"mean": float(vals.mean()) if len(vals) else float("nan"),
                  "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
    return out


def _cross_validate(config: dict, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """5 折说话人 CV：默认权重 / 训练折选权重 / 岭回归 / 说话人级基线 / 校准关。"""
    from src.fusion.learned_fusion import fit_from_rows
    folds = _cv_folds(rows)
    res: dict[str, list[dict[str, float]]] = defaultdict(list)
    heldout_predictions: list[dict[str, Any]] = []
    cfg_nocal = {**config, "fusion_calibration": {"enabled": False}}
    for held in folds:
        test = [r for r in rows if r["speaker"] in held]
        train = [r for r in rows if r["speaker"] not in held]
        # (0) 校准关 + 默认权重
        res["no_calibration"].append(_metrics_of(_apply(cfg_nocal, test)))
        # (a) 语料校准 + 默认权重（留出折预测同时用于不确定性校准）
        pred_default = _apply(config, test)
        res["default"].append(_metrics_of(pred_default))
        heldout_predictions.extend(pred_default)
        # (b) 训练折上网格选权重
        best = max(_weight_grid(config["fusion_weights"]),
                   key=lambda nw: _score_block(_apply(config, train, weights=nw[1]))["composite"])
        res["grid_on_train"].append({**_metrics_of(_apply(config, test, weights=best[1])), "chosen": best[0]})
        # (c) 岭回归（训练折拟合，原始分）
        try:
            lf = fit_from_rows(train, l2=1.0)
            res["learned"].append(_metrics_of(_apply(config, test, learned=lf)))
        except ValueError:
            pass
        # (d) 说话人级中性基线（每位留出说话人用自己的 neutral 句算偏移）
        sp_rows: list[dict[str, Any]] = []
        for spk in held:
            off = _speaker_offsets(rows, spk)
            base = WeightedFusionOffsets(config)
            merged = {**base, **off} if off else base
            sp_rows.extend(_apply(config, [r for r in test if r["speaker"] == spk], offsets=merged))
        res["speaker_baseline"].append(_metrics_of(sp_rows))
    out = {name: _agg(v) for name, v in res.items()}
    out["folds_speakers"] = folds
    out["uncertainty_calibration"] = _uncertainty_calibration(heldout_predictions)
    return out


def _uncertainty_calibration(preds: list[dict[str, Any]]) -> dict[str, Any]:
    """留出折预测上：分歧度（max(negative_sd, arousal_sd)）三等分，每档象限准确率。"""
    from scipy.stats import spearmanr
    labeled = [p for p in preds if p["emotion"] in EMOTION_TO_QUADRANT and p.get("uncertainty")]
    if len(labeled) < 30:
        return {}
    score = np.array([max(p["uncertainty"].get("negative_sd", 0.0), p["uncertainty"].get("arousal_sd", 0.0))
                      for p in labeled])
    correct = np.array([p["dominant_quadrant"] == EMOTION_TO_QUADRANT[p["emotion"]] for p in labeled], dtype=float)
    c1, c2 = np.percentile(score, [100 / 3, 200 / 3])
    masks = {"t0": score < c1, "t1": (score >= c1) & (score < c2), "t2": score >= c2}
    terciles: dict[str, dict[str, Any]] = {}
    for t, mask in masks.items():
        terciles[t] = {"n": int(mask.sum()),
                       "accuracy": float(correct[mask].mean()) if mask.any() else float("nan"),
                       "score_range": [float(score[mask].min()) if mask.any() else None,
                                       float(score[mask].max()) if mask.any() else None]}
    # 等级按实测准确率排序命名，不假设「分歧小 = 可信」（实测方向相反，见 reliability.py）
    order = sorted(terciles, key=lambda t: -terciles[t]["accuracy"])
    for t, g in zip(order, ("high", "medium", "low"), strict=False):
        terciles[t]["grade"] = g
    rho = spearmanr(score, correct).correlation
    return {"cuts": [round(float(c1), 4), round(float(c2), 4)], "terciles": terciles,
            "spearman_score_vs_correct": float(rho) if np.isfinite(rho) else None,
            "n": len(labeled),
            "definition": "score = max(negative_sd, arousal_sd) of active modalities; terciles on held-out CV "
                          "predictions; grade assigned by empirical accuracy rank (not by assumed direction)"}


def WeightedFusionOffsets(config: dict) -> dict[str, tuple[float, float]]:
    """当前配置下的语料偏移（供说话人级基线在其上覆盖）。"""
    from src.fusion.weighted_fusion import load_modality_calibration
    return load_modality_calibration(config)


def _emotion_summarize(config: dict, rows: list[dict[str, Any]]) -> int:
    import jiwer

    from src.fusion.weighted_fusion import MODALITIES
    summary: dict[str, Any] = {"n": len(rows),
                               "emotions": dict(Counter(r["emotion"] for r in rows)),
                               "speakers": len({r["speaker"] for r in rows})}
    dev, test, dev_spk, test_spk = _split_dev_test(rows)
    summary["split"] = {"dev_speakers": dev_spk, "test_speakers": test_spk,
                        "n_dev": len(dev), "n_test": len(test)}
    # 校准开/关（全体，用于展示校准本身的作用）
    cfg_nocal = {**config, "fusion_calibration": {"enabled": False}}
    summary["calibration_off"] = _score_block(_apply(cfg_nocal, rows))
    summary["calibration_on"] = _score_block(_apply(config, rows))
    # 权重网格：只看 dev
    grid_rows = []
    for name, w in _weight_grid(config["fusion_weights"]):
        b_dev = _score_block(_apply(config, dev, weights=w))
        b_test = _score_block(_apply(config, test, weights=w))
        grid_rows.append({"name": name, "weights": w,
                          "dev": {"accuracy": b_dev["quadrant"]["accuracy"], "rhoA": b_dev["direction"]["spearman_arousal"],
                                  "rhoV": b_dev["direction"]["spearman_valence"], "composite": b_dev["composite"]},
                          "test": {"accuracy": b_test["quadrant"]["accuracy"], "rhoA": b_test["direction"]["spearman_arousal"],
                                   "rhoV": b_test["direction"]["spearman_valence"], "composite": b_test["composite"]}})
    best = max(grid_rows, key=lambda g: g["dev"]["composite"])
    summary["weight_grid"] = grid_rows
    summary["chosen_weights"] = {"name": best["name"], "weights": best["weights"]}
    # 用 dev 选出的权重在 test 上报告（正式数字）
    test_rows = _apply(config, test, weights=best["weights"])
    summary["test"] = _score_block(test_rows)
    summary["dev"] = _score_block(_apply(config, dev, weights=best["weights"]))
    # 全体（用选定权重）：与 v0.1 报告口径一致的整体表
    rows = _apply(config, rows, weights=best["weights"])
    config = {**config, "fusion_weights": best["weights"]}
    summary["full"] = {"quadrant": _quadrant_metrics(rows), "direction": _direction_metrics(rows)}
    # ---- v0.3：性别均衡 5 折说话人交叉验证 ----
    summary["cv"] = _cross_validate(config, rows)
    unc = summary["cv"].get("uncertainty_calibration") or {}
    if unc:
        _json_dump({"_meta": {"source": "CSEMOTIONS 5-fold speaker CV held-out predictions, default config",
                              "generated": date.today().isoformat(), "script": "scripts/evaluate.py emotion",
                              "n": unc["n"], "definition": unc["definition"],
                              "spearman_score_vs_correct": unc["spearman_score_vs_correct"]},
                    "score": "max(negative_sd, arousal_sd)", "cuts": unc["cuts"], "terciles": unc["terciles"]}, UNC_PATH)
    # ASR 后验置信度：分布及其与逐句 CER 的相关
    summary["asr_confidence"] = _asr_confidence_analysis(rows)
    # 用全部数据拟合可学习融合并写出（默认不启用）
    try:
        from src.fusion.learned_fusion import fit_from_rows
        lf = fit_from_rows(rows, l2=1.0, meta={
            "source": "CSEMOTIONS full sample (acted emotions, studio)",
            "cv": summary["cv"]["learned"], "generated": date.today().isoformat(),
            "script": "scripts/evaluate.py emotion"})
        lf.save(LEARNED_PATH)
        summary["learned_fusion_path"] = str(LEARNED_PATH)
    except ValueError as e:
        summary["learned_fusion_error"] = str(e)
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
    t = summary["test"]
    _log(f"[全体，权重={summary['chosen_weights']['name']}] 象限准确率={q['accuracy']:.3f} (n={q['n']})  "
         f"ρA={d['spearman_arousal']:.3f} ρV={d['spearman_valence']:.3f}  方向 {d['pairwise_pass']}/{d['pairwise_total']}")
    _log(f"[test 说话人 {summary['split']['test_speakers']}] 象限准确率={t['quadrant']['accuracy']:.3f} "
         f"ρA={t['direction']['spearman_arousal']:.3f} ρV={t['direction']['spearman_valence']:.3f}")
    _log(f"[校准 off→on，全体] 准确率 {summary['calibration_off']['quadrant']['accuracy']:.3f} → "
         f"{summary['calibration_on']['quadrant']['accuracy']:.3f}")
    cv = summary["cv"]
    for name in ("no_calibration", "default", "grid_on_train", "learned", "speaker_baseline"):
        if name in cv:
            c = cv[name]
            _log(f"[5 折 CV] {name:16s} acc={c['accuracy']['mean']:.3f}±{c['accuracy']['sd']:.3f} "
                 f"ρA={c['rhoA']['mean']:.2f}±{c['rhoA']['sd']:.2f} ρV={c['rhoV']['mean']:.2f}±{c['rhoV']['sd']:.2f}")
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


# ---------------------------------------------------------------------- #
# sequence（v0.5）：用同一说话人同一情绪的连续句子模拟会话
# ---------------------------------------------------------------------- #
def cmd_sequence(args: argparse.Namespace) -> int:
    """在缓存的 CSEMOTIONS 留出折预测上，把每个 (说话人, 情绪) 的句子按顺序当作一次会话，
    比较「每段独立判定」与「会话平滑 + 滞回」的准确率、翻转次数与首次正确所需段数。
    不跑模型，只用缓存。"""
    from src.config_loader import load_settings
    from src.session.state_tracker import StateTracker, TrackerConfig

    rows = _json_load(RESULTS_DIR / "emotion_rows.json")
    config = load_settings()
    # 留出折预测（默认配置），保证与正式数字同一口径
    preds: list[dict[str, Any]] = []
    for held in _cv_folds(rows):
        preds += _apply(config, [r for r in rows if r["speaker"] in held])
    seqs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for p in preds:
        if p["emotion"] in EMOTION_TO_QUADRANT:
            seqs[(p["speaker"], p["emotion"])].append(p)
    for k in seqs:
        seqs[k].sort(key=lambda p: p["wav"])

    base = TrackerConfig.from_settings(config)
    variants = {
        "independent": None,
        f"ema{base.ema_alpha:g}_k{base.min_consecutive}（默认）": base,
        "ema1.0_k1（无平滑无滞回）": TrackerConfig(1.0, base.hysteresis_band, 1, base.mid_v, base.mid_a, base.quadrant_band),
        "ema0.5_k1（仅平滑）": TrackerConfig(0.5, base.hysteresis_band, 1, base.mid_v, base.mid_a, base.quadrant_band),
        "ema1.0_k2（仅滞回）": TrackerConfig(1.0, base.hysteresis_band, 2, base.mid_v, base.mid_a, base.quadrant_band),
        "ema0.3_k2（更平稳）": TrackerConfig(0.3, base.hysteresis_band, 2, base.mid_v, base.mid_a, base.quadrant_band),
    }
    out: dict[str, Any] = {"n_sequences": len(seqs), "seq_len": Counter(len(v) for v in seqs.values()).most_common(1)[0][0],
                           "variants": {}}
    for name, tc in variants.items():
        per_pos_correct: dict[int, list[float]] = defaultdict(list)
        final_correct, flips, first_correct_pos = [], [], []
        for (spk, emo), seq in seqs.items():
            ref = EMOTION_TO_QUADRANT[emo]
            tracker = StateTracker(tc) if tc is not None else None
            first = None
            for i, p in enumerate(seq, 1):
                if tracker is None:
                    q = p["dominant_quadrant"]
                else:
                    q = tracker.update(p["negative"], p["arousal"])["quadrant"]
                ok = float(q == ref)
                per_pos_correct[i].append(ok)
                if ok and first is None:
                    first = i
            final_correct.append(per_pos_correct[len(seq)][-1])
            if tracker is not None:
                flips.append(tracker.state()["flips"])
            else:
                qs = [p["dominant_quadrant"] for p in seq]
                flips.append(sum(1 for a, b in zip(qs, qs[1:], strict=False) if a != b))
            first_correct_pos.append(first if first is not None else len(seq) + 1)
        out["variants"][name] = {
            "accuracy_by_position": {str(i): float(np.mean(v)) for i, v in sorted(per_pos_correct.items())},
            "accuracy_final": float(np.mean(final_correct)),
            "accuracy_all_positions": float(np.mean([x for v in per_pos_correct.values() for x in v])),
            "mean_flips_per_sequence": float(np.mean(flips)),
            "sequences_with_any_flip": float(np.mean([f > 0 for f in flips])),
            "median_first_correct_position": float(np.median(first_correct_pos)),
            "never_correct_fraction": float(np.mean([p > out["seq_len"] for p in first_correct_pos])),
        }
    _json_dump(out, RESULTS_DIR / "sequence_summary.json")
    for name, v in out["variants"].items():
        _log(f"{name:28s} 末段准确率={v['accuracy_final']:.3f}  全位置={v['accuracy_all_positions']:.3f}  "
             f"平均翻转={v['mean_flips_per_sequence']:.2f}  有翻转序列占比={v['sequences_with_any_flip']:.2f}  "
             f"首次正确中位位置={v['median_first_correct_position']:.0f}")
    return 0


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
        if s.get("asr_confidence"):
            a = s["asr_confidence"]
            out.append(f"| ASR 置信度（{a['source']}）均值 / P5 / 最小 | {_fmt(a['mean'])} / {_fmt(a['p5'])} / {_fmt(a['min'])} |")
            out.append(f"| 置信度 vs 逐句 CER 的 Spearman ρ | {_fmt(a['spearman_conf_vs_cer'])} |")
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
        if s.get("asr_confidence"):
            a = s["asr_confidence"]
            out.append(f"**ASR 置信度（{a['source']}，n={a['n']}）**：均值 {_fmt(a['mean'])}，SD {_fmt(a['sd'])}，"
                       f"P5 {_fmt(a['p5'])}，P25 {_fmt(a['p25'])}，最小 {_fmt(a['min'])}；与逐句 CER 的 Spearman ρ = "
                       f"{_fmt(a['spearman_conf_vs_cer'])}；置信度最低 10% 句子的 CER {_fmt(a['cer_when_conf_below_p10'])} vs "
                       f"其余 {_fmt(a['cer_when_conf_above_p10'])}\n")
        if "cv" in s and s["cv"].get("uncertainty_calibration"):
            u = s["cv"]["uncertainty_calibration"]
            out.append(f"**不确定性校准**（留出折预测 n={u['n']}；分歧度 = 活跃模态加权 SD 的较大者；"
                       f"分歧度与判对的 Spearman ρ = {_fmt(u['spearman_score_vs_correct'])}，**正相关**：模态一致往往"
                       f"意味着都接近中性、证据弱；等级按各档实测准确率命名）\n")
            out.append("| 分歧度三分位 | 区间 | n | 象限准确率 | 等级 |\n|------|------|--:|---:|------|")
            for t in ("t0", "t1", "t2"):
                b = u["terciles"][t]
                out.append(f"| {t} | {_fmt(b['score_range'][0])}–{_fmt(b['score_range'][1])} | {b['n']} | {_fmt(b['accuracy'])} | {b.get('grade', '—')} |")
            out.append("")
        if "cv" in s:
            cv = s["cv"]
            out.append(f"**性别均衡 5 折说话人交叉验证**（折：{cv['folds_speakers']}；均值 ± 标准差；正式数字）\n")
            out.append("| 设置 | 象限准确率 | ρ arousal | ρ valence | 方向检验(8) |\n|------|---:|---:|---:|---:|")
            labels = {"no_calibration": "校准关 + 默认权重", "default": "语料校准 + 默认权重",
                      "grid_on_train": "语料校准 + 训练折选权重", "learned": "岭回归（训练折拟合）",
                      "speaker_baseline": "说话人级中性基线 + 默认权重"}
            for key, lab in labels.items():
                if key in cv:
                    c = cv[key]
                    out.append(f"| {lab} | {c['accuracy']['mean']:.3f} ± {c['accuracy']['sd']:.3f} | "
                               f"{c['rhoA']['mean']:.2f} ± {c['rhoA']['sd']:.2f} | {c['rhoV']['mean']:.2f} ± {c['rhoV']['sd']:.2f} | "
                               f"{c['pairwise_pass']['mean']:.1f} |")
            out.append("")
        if "split" in s:
            sp = s["split"]
            out.append("**中性校准（全体样本）**\n")
            out.append("| 设置 | 象限准确率 | ρ arousal | ρ valence |\n|------|---:|---:|---:|")
            for lab, key in (("校准关", "calibration_off"), ("校准开", "calibration_on")):
                b = s[key]
                out.append(f"| {lab} | {_fmt(b['quadrant']['accuracy'])} | {_fmt(b['direction']['spearman_arousal'])} | {_fmt(b['direction']['spearman_valence'])} |")
            out.append("")
            out.append(f"**融合权重网格（仅在 dev 说话人 {sp['dev_speakers']} 上选择，n={sp['n_dev']}；"
                       f"test 说话人 {sp['test_speakers']}，n={sp['n_test']}）**\n")
            out.append("| 候选 | dev 准确率 | dev ρA | dev ρV | dev 综合 | test 准确率 | test ρA | test ρV |\n|------|---:|---:|---:|---:|---:|---:|---:|")
            for g in s["weight_grid"]:
                mark = " **←选定**" if g["name"] == s["chosen_weights"]["name"] else ""
                out.append(f"| {g['name']}{mark} | {_fmt(g['dev']['accuracy'])} | {_fmt(g['dev']['rhoA'])} | {_fmt(g['dev']['rhoV'])} | {_fmt(g['dev']['composite'])} | "
                           f"{_fmt(g['test']['accuracy'])} | {_fmt(g['test']['rhoA'])} | {_fmt(g['test']['rhoV'])} |")
            out.append("")
            t = s["test"]
            out.append(f"**正式数字（test 说话人，选定权重）**：象限准确率 **{_fmt(t['quadrant']['accuracy'])}**（n={t['quadrant']['n']}），"
                       f"ρA {_fmt(t['direction']['spearman_arousal'])}，ρV {_fmt(t['direction']['spearman_valence'])}，"
                       f"方向检验 {t['direction']['pairwise_pass']}/{t['direction']['pairwise_total']}\n")
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
    p = RESULTS_DIR / "sequence_summary.json"
    if p.exists():
        s = _json_load(p)
        out.append(f"### v0.5：会话内平滑 / 滞回的序列模拟（CSEMOTIONS，{s['n_sequences']} 个说话人×情绪序列，"
                   f"每序列 {s['seq_len']} 段，留出折预测）\n")
        out.append("| 设置 | 末段准确率 | 全位置准确率 | 每序列平均翻转 | 有翻转的序列占比 | 首次正确中位位置 | 从未正确占比 |\n"
                   "|------|---:|---:|---:|---:|---:|---:|")
        for name, v in s["variants"].items():
            out.append(f"| {name} | {_fmt(v['accuracy_final'])} | {_fmt(v['accuracy_all_positions'])} | "
                       f"{_fmt(v['mean_flips_per_sequence'], 2)} | {_fmt(v['sequences_with_any_flip'], 2)} | "
                       f"{v['median_first_correct_position']:.0f} | {_fmt(v['never_correct_fraction'], 2)} |")
        out.append("")
        out.append("按位置的准确率（默认设置 vs 独立判定）：\n")
        names = list(s["variants"])
        default_name = next(n for n in names if "默认" in n)
        out.append("| 位置 | 独立判定 | " + default_name + " |\n|---:|---:|---:|")
        for i in s["variants"]["independent"]["accuracy_by_position"]:
            out.append(f"| {i} | {_fmt(s['variants']['independent']['accuracy_by_position'][i])} | "
                       f"{_fmt(s['variants'][default_name]['accuracy_by_position'][i])} |")
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
    p = sub.add_parser("neutral", help="AISHELL-3 全管线：CER + 中性语音输出分布 + 写中性校准偏移")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--no-write-calibration", action="store_true",
                   help="不写 config/modality_calibration.json（仅报告）")
    p = sub.add_parser("emotion", help="CSEMOTIONS 情绪判别验证")
    p.add_argument("--per-cell", type=int, default=6, help="每（情绪×说话人）抽样句数")
    p = sub.add_parser("resummarize", help="仅在缓存上重算 emotion 汇总（不跑模型）")
    sub.add_parser("sequence", help="v0.5：会话内平滑/滞回的离线序列模拟（用缓存，不跑模型）")
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
    if args.cmd == "sequence":
        return cmd_sequence(args)
    if args.cmd == "resummarize":
        from src.config_loader import load_settings
        return _emotion_summarize(load_settings(), _json_load(RESULTS_DIR / "emotion_rows.json"))
    return cmd_report(args)


if __name__ == "__main__":
    sys.exit(main())
