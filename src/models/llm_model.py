"""Qwen3-1.7B 文本语义情感 LLM 封装（bitsandbytes NF4 量化）。

通过 prompt 让模型输出两个 0~1 的浮点数（负面分、唤醒度），用正则解析。
解析失败时 temperature=0 重试一次，二次失败降级为文本统计分数。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.config_loader import load_settings

logger = logging.getLogger("mandarin_emo_stim.llm")

_PROMPT_TEMPLATE = (
    "请分析以下中文口语转写文本的情绪，只输出两个0到1之间的小数"
    "（空格分隔，保留两位小数），第一个是负面情绪分"
    "（0=极度正面，1=极度负面），第二个是情绪唤醒度"
    "（0=极度平静，1=极度激动）。不要输出任何其他内容。\n"
    "/no_think\n\n文本：{asr_text}"
)

_FLOAT_PAIR_RE = re.compile(r"([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)")


class LLMModel:
    """Qwen3-1.7B 文本语义情感分析封装。"""

    def __init__(self, device: str = "cuda", model: Any = None, tokenizer: Any = None):
        self.device = device
        if model is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            self._load()
        logger.info("LLM 模型就绪（device=%s）", self.device)

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        settings = load_settings()["models"]
        model_name = settings["llm_model"]
        logger.info("加载 LLM: %s（NF4 4bit）", model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if self.device.startswith("cuda") and _bitsandbytes_available():
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            # 纯 CPU 模式：FP16 推理（bitsandbytes 不可用）
            logger.info("bitsandbytes 不可用或 CPU 模式，使用 FP16 加载")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).to(self.device)

    def analyze_text(self, asr_text: str) -> dict[str, Any]:
        """分析文本情绪，输出负面分与唤醒度。

        Returns:
            ``{"s_text_llm": float, "a_text_llm": float, "raw": str, "fallback": bool}``。
            解析失败时降级为 (0.5, 0.5) 并标记 fallback。
        """
        if not asr_text or not asr_text.strip():
            return {"s_text_llm": 0.5, "a_text_llm": 0.5, "raw": "", "fallback": True}

        # 第一次：temperature=0.1
        raw = self._generate(asr_text, temperature=0.1)
        parsed = self._parse(raw)
        if parsed is not None:
            s, a = parsed
            return {"s_text_llm": s, "a_text_llm": a, "raw": raw, "fallback": False}

        # 重试：temperature=0
        logger.warning("LLM 输出解析失败，重试（temperature=0）：%s", raw)
        raw = self._generate(asr_text, temperature=0.0)
        parsed = self._parse(raw)
        if parsed is not None:
            s, a = parsed
            return {"s_text_llm": s, "a_text_llm": a, "raw": raw, "fallback": False}

        # 二次失败：降级
        logger.warning("LLM 二次解析失败，降级为中性分：%s", raw)
        return {"s_text_llm": 0.5, "a_text_llm": 0.5, "raw": raw, "fallback": True}

    def _generate(self, asr_text: str, temperature: float) -> str:
        import torch
        settings = load_settings()["models"]
        prompt = _PROMPT_TEMPLATE.format(asr_text=asr_text)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=int(settings.get("max_new_tokens", 16)),
                do_sample=temperature > 0,
                temperature=max(temperature, 0.01) if temperature > 0 else 1.0,
                top_p=0.9,
                repetition_penalty=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # 仅取新生成的部分
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @staticmethod
    def _parse(text: str) -> tuple[float, float] | None:
        m = _FLOAT_PAIR_RE.search(text)
        if not m:
            return None
        try:
            s = float(m.group(1))
            a = float(m.group(2))
        except ValueError:
            return None
        s = max(0.0, min(1.0, s))
        a = max(0.0, min(1.0, a))
        return s, a


def _bitsandbytes_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        return True
    except Exception:
        return False
