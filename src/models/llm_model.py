"""Qwen3-1.7B 文本语义情感 LLM 封装（bitsandbytes NF4 量化）。

通过 few-shot prompt 让模型输出两个 0~1 的浮点数（负面分、唤醒度），用正则解析。
启用 ``enable_thinking=False`` 跳过 Qwen3 的思考模式（避免 <think> 块干扰）。
解析失败时 temperature=0 重试一次，二次失败降级为文本统计分数。

注：1.7B 小模型对精确数值评分能力有限，故采用 few-shot 示例约束输出格式与量纲，
实际情感量化以 6 模态加权融合为主，LLM 仅作为文本语义支路之一。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.config_loader import load_settings

logger = logging.getLogger("mandarin_emo_stim.llm")

_SYSTEM_PROMPT = (
    "你是情绪分析器。对中文句子输出两个0到1的小数（空格分隔）："
    "第一个是负面情绪分（0=非常正面，1=非常负面），"
    "第二个是情绪唤醒度（0=非常平静，1=非常激动）。只输出两个数字，不要解释。"
)

# few-shot 示例（约束输出量纲，提升小模型一致性）
_FEW_SHOTS = [
    ("我今天非常开心！", "0.10 0.80"),
    ("我很难过，太痛苦了", "0.90 0.30"),
    ("今天天气不错", "0.40 0.40"),
    ("气死我了，太过分了！", "0.95 0.95"),
]

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

        # 重试：temperature=0（贪婪）
        logger.warning("LLM 输出解析失败，重试（temperature=0）：%s", raw)
        raw = self._generate(asr_text, temperature=0.0)
        parsed = self._parse(raw)
        if parsed is not None:
            s, a = parsed
            return {"s_text_llm": s, "a_text_llm": a, "raw": raw, "fallback": False}

        # 二次失败：降级
        logger.warning("LLM 二次解析失败，降级为中性分：%s", raw)
        return {"s_text_llm": 0.5, "a_text_llm": 0.5, "raw": raw, "fallback": True}

    def _build_messages(self, asr_text: str) -> list[dict]:
        """构建 few-shot 消息序列。"""
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for text, answer in _FEW_SHOTS:
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": asr_text})
        return messages

    def _generate(self, asr_text: str, temperature: float) -> str:
        import torch
        settings = load_settings()["models"]
        messages = self._build_messages(asr_text)
        # enable_thinking=False 跳过 Qwen3 思考模式
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # 旧版 transformers 不支持 enable_thinking 参数
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
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
