"""无头分析 CLI（不启动 GUI）。

用法::

    python -m src.cli --audio path/to/test.wav [--out output.wav] [--duration 30]

加载模型 → 分析音频 → 打印量化指标 → 生成并保存声刺激 WAV。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# 触发便携环境重定向
from src import portable  # noqa: F401

logger = logging.getLogger("mandarin_emo_stim.cli")
from src.models.model_manager import ModelManager
from src.pipeline import AnalysisPipeline
from src.stimulus.generator import StimulusGenerator


def _progress(stage: str, pct: int) -> None:
    print(f"  [{pct:3d}%] {stage}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mandarin-EmoStim 无头分析")
    parser.add_argument("--audio", required=False, default=None, help="待分析的音频文件路径")
    parser.add_argument("--out", default=None, help="生成的刺激 WAV 输出路径")
    parser.add_argument("--duration", type=float, default=None,
                        help="刺激时长（秒），默认取配置")
    parser.add_argument("--json", default=None, help="把完整结果写入该 JSON 文件")
    parser.add_argument("--calibrate-user", action="store_true",
                        help="把 --audio 当作用户的平静朗读样本，计算并保存个人基线（不生成刺激）")
    parser.add_argument("--clear-user-calibration", action="store_true", help="删除已保存的个人基线")
    args = parser.parse_args(argv)

    if args.clear_user_calibration:
        from src.fusion import personal_calibration
        print("已删除个人基线" if personal_calibration.clear() else "没有个人基线可删除", flush=True)
        if not args.calibrate_user and args.audio in (None, ""):
            return 0

    # 先做廉价的输入校验，再加载耗时的模型（也避免测试环境无模型时触发下载）
    if args.audio is None:
        print("\n[错误] 需要 --audio", flush=True)
        return 2
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"\n[错误] 文件不存在：{audio_path}", flush=True)
        return 1
    if audio_path.suffix.lower() not in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
        print(f"\n[错误] 不支持的音频格式：{audio_path.suffix}", flush=True)
        return 1

    manager = None
    try:
        print("=== 加载模型 ===", flush=True)
        manager = ModelManager()
        manager.load_all(progress_cb=_progress)

        print("=== 分析音频 ===", flush=True)
        pipeline = AnalysisPipeline(manager)
        result = pipeline.analyze(args.audio, progress_cb=_progress)

        if args.calibrate_user:
            from src.fusion import personal_calibration
            offsets = personal_calibration.compute_offsets([result["modal_scores_raw"]])
            path = personal_calibration.save(offsets, meta={"source": str(audio_path),
                                                            "duration_sec": result["duration"]})
            print(f"\n=== 个人基线已保存: {path} ===", flush=True)
            for m, v in offsets.items():
                print(f"  {m:10s} negative {v['negative']:+.3f}  arousal {v['arousal']:+.3f}", flush=True)
            manager.unload_all()
            return 0

        print("\n=== 量化指标 ===", flush=True)
        print(f"  Negative Score : {result['negative']:.3f}")
        print(f"  Valence        : {result['valence']:.3f}")
        print(f"  Arousal        : {result['arousal']:.3f}")
        print(f"  主象限         : {result['dominant_quadrant']}")
        print(f"  ASR 文本       : {result['asr_text']}")
        print(f"  音频质量 SNR   : {result['audio_quality']['snr_db']:.1f} dB")
        print(f"  有效时长       : {result['duration']:.2f} s")
        unc = result.get("uncertainty") or {}
        if unc:
            print(f"  模态分歧(SD)   : negative {unc.get('negative_sd', 0):.3f}  "
                  f"arousal {unc.get('arousal_sd', 0):.3f}  [{result.get('calibration_source', 'none')} 校准, "
                  f"{result.get('fusion_mode', 'weighted')} 融合]")
        if result.get("degraded_modalities"):
            print(f"  [警告] 以下模态因异常降级为中性分: "
                  f"{', '.join(result['degraded_modalities'])}", flush=True)

        print("\n=== 生成声刺激 ===", flush=True)
        generator = StimulusGenerator()
        stim = generator.generate(
            result["valence"], result["arousal"],
            memberships=result["memberships"], duration=args.duration,
        )
        out_path = args.out or str(portable.HISTORY_STIMULI_DIR / "cli_output.wav")
        portable.HISTORY_STIMULI_DIR.mkdir(parents=True, exist_ok=True)
        import soundfile as sf
        sf.write(out_path, stim, generator.sr, subtype="PCM_16")
        print(f"  刺激音频已保存: {out_path}", flush=True)

        if args.json:
            serializable = {k: v for k, v in result.items()
                            if not isinstance(v, (bytes,))}
            Path(args.json).write_text(
                json.dumps(serializable, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            print(f"  结果已写入: {args.json}", flush=True)

        if manager is not None:
            manager.unload_all()
        return 0

    except KeyboardInterrupt:
        print("\n[中断] 用户中断（Ctrl+C），正在退出…", flush=True)
        if manager is not None:
            manager.unload_all()
        return 130  # Unix 惯例：128+SIGINT(2)
    except FileNotFoundError as e:
        print(f"\n[错误] 文件不存在：{e}", flush=True)
        return 1
    except Exception as e:  # noqa: BLE001
        logger.exception("CLI 运行失败")
        print(f"\n[错误] {type(e).__name__}: {e}", flush=True)
        if manager is not None:
            manager.unload_all()
        return 1


if __name__ == "__main__":
    sys.exit(main())
