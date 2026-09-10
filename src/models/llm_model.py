"""Qwen3-1.7B 文本语义情感 LLM 封装（bitsandbytes NF4 量化）。

通过 few-shot prompt 让模型输出两个 0~1 的浮点数（负面分、唤醒度），用正则解析。
启用 ``enable_thinking=False`` 跳过 Qwen3 的思考模式（避免 <think> 块干扰）。
**首次调用使用 greedy 解码（do_sample=False）以保证同一输入结果可复现**；
仅当解析失败时才以低温采样重试一次，二次失败降级为文本统计分数。

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
    "你是情绪评分器。对给定的中文句子输出两个 0 到 1 的小数（空格分隔）："
    "第一个是负面分（0=非常正面，0.5=没有明显情绪的中性陈述，1=非常负面），"
    "第二个是唤醒度（0=非常平静，0.5=一般陈述，1=非常激动）。"
    "平铺直叙、只陈述事实的句子两个值都应接近 0.5，不要因为句子没有褒义词就判为负面。"
    "只输出两个数字，不要解释。"
)

# few-shot 示例（约束输出量纲；v0.2 加入中性陈述与平静正面例子，
# 修正 v0.1 对中性文本系统性偏负的问题——AISHELL-3 中性语音实测负面分 0.66）
_FEW_SHOTS = [
    ("我今天非常开心！", "0.10 0.80"),
    ("会议安排在下午三点，请大家准时参加。", "0.50 0.45"),
    ("我很难过，太痛苦了", "0.90 0.35"),
    ("这个城市的地铁一共有十二条线路。", "0.50 0.40"),
    ("气死我了，太过分了！", "0.95 0.95"),
    ("阳光洒在湖面上，我们安静地坐着喝茶。", "0.30 0.25"),
    ("独自在空荡荡的停车场，我心里发毛。", "0.75 0.70"),
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
            # 无 bitsandbytes / 非 CUDA：CPU 上用 float32（PyTorch CPU 的 fp16 矩阵乘不支持或极慢），
            # 其他设备用 fp16
            dtype = torch.float32 if not self.device.startswith("cuda") else torch.float16
            logger.info("bitsandbytes 不可用或非 CUDA 设备，使用 %s 加载", dtype)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
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

        # 第一次：greedy（确定性，可复现）
        raw = self._generate(asr_text, temperature=0.0)
        parsed = self._parse(raw)
        if parsed is not None:
            s, a = parsed
            return {"s_text_llm": s, "a_text_llm": a, "raw": raw, "fallback": False}

        # 重试：低温采样，跳出 greedy 的固定坏输出
        logger.warning("LLM 输出解析失败，低温采样重试：%s", raw)
        raw = self._generate(asr_text, temperature=0.3)
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
