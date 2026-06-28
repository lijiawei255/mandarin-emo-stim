"""无头分析 CLI（不启动 GUI）。

用法::

    python -m src.cli --audio path/to/test.wav [--out output.wav] [--duration 30]

加载模型 → 分析音频 → 打印量化指标 → 生成并保存声刺激 WAV。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 触发便携环境重定向
from src import portable  # noqa: F401
from src.models.model_manager import ModelManager
from src.pipeline import AnalysisPipeline
from src.stimulus.generator import StimulusGenerator


def _progress(stage: str, pct: int) -> None:
    print(f"  [{pct:3d}%] {stage}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mandarin-EmoStim 无头分析")
    parser.add_argument("--audio", required=True, help="待分析的音频文件路径")
    parser.add_argument("--out", default=None, help="生成的刺激 WAV 输出路径")
    parser.add_argument("--duration", type=float, default=None,
                        help="刺激时长（秒），默认取配置")
    parser.add_argument("--json", default=None, help="把完整结果写入该 JSON 文件")
    args = parser.parse_args(argv)

    print("=== 加载模型 ===", flush=True)
    manager = ModelManager()
    manager.load_all(progress_cb=_progress)

    print("=== 分析音频 ===", flush=True)
    pipeline = AnalysisPipeline(manager)
    result = pipeline.analyze(args.audio, progress_cb=_progress)

    print("\n=== 量化指标 ===", flush=True)
    print(f"  Negative Score : {result['negative']:.3f}")
    print(f"  Valence        : {result['valence']:.3f}")
    print(f"  Arousal        : {result['arousal']:.3f}")
    print(f"  主象限         : {result['dominant_quadrant']}")
    print(f"  ASR 文本       : {result['asr_text']}")
    print(f"  音频质量 SNR   : {result['audio_quality']['snr_db']:.1f} dB")
    print(f"  有效时长       : {result['duration']:.2f} s")

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

    manager.unload_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
