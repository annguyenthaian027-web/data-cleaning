"""Lớp trừu tượng nhà cung cấp LLM (Claude / OpenAI). SDK được import lười: không cài SDK nào vẫn chạy được chế độ ngoại tuyến."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional, Protocol

from .config import DEFAULT_MODELS, AIConfig


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    data: dict
    provider: str
    model: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: int = 0


class LLMProvider(Protocol):
    name: str
    model: str

    def generate_json(self, *, system: str, user: str, schema: dict, tool_name: str, tool_description: str,
                      max_tokens: int) -> LLMResult: ...


class AnthropicProvider:
    """Claude: ép trả JSON bằng tool-use (tool_choice cố định) — an toàn hơn là nhờ model 'trả JSON' trong văn bản."""
    name = "anthropic"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, timeout_s: float = 60.0,
                 max_retries: int = 2, client=None):
        self.model = model or DEFAULT_MODELS["anthropic"]
        if client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover
                raise LLMError("Chưa cài SDK: pip install anthropic") from e
            client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s, max_retries=max_retries)
        self.client = client

    def generate_json(self, *, system, user, schema, tool_name, tool_description, max_tokens) -> LLMResult:
        t0 = time.perf_counter()
        try:
            msg = self.client.messages.create(
                model=self.model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}],
                tools=[{"name": tool_name, "description": tool_description, "input_schema": schema}],
                tool_choice={"type": "tool", "name": tool_name})
        except Exception as e:  # noqa: BLE001 - SDK ném nhiều loại; gói lại một kiểu duy nhất cho tầng trên
            raise LLMError(f"Lỗi gọi Claude API: {type(e).__name__}: {e}") from e
        if getattr(msg, "stop_reason", None) == "max_tokens":
            raise LLMError("Phản hồi bị cắt do hết max_tokens.")
        block = next((b for b in msg.content if getattr(b, "type", None) == "tool_use"), None)
        if block is None or not isinstance(block.input, dict):
            raise LLMError("Claude không trả về kết quả có cấu trúc (tool_use).")
        u = getattr(msg, "usage", None)
        return LLMResult(block.input, self.name, getattr(msg, "model", self.model),
                         getattr(u, "input_tokens", None), getattr(u, "output_tokens", None),
                         int((time.perf_counter() - t0) * 1000))


class OpenAIProvider:
    """OpenAI (hoặc endpoint tương thích qua base_url): dùng response_format=json_schema."""
    name = "openai"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, timeout_s: float = 60.0,
                 max_retries: int = 2, base_url: Optional[str] = None, client=None):
        self.model = model or DEFAULT_MODELS["openai"]
        if client is None:
            try:
                import openai
            except ImportError as e:  # pragma: no cover
                raise LLMError("Chưa cài SDK: pip install openai") from e
            client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=max_retries)
        self.client = client

    def generate_json(self, *, system, user, schema, tool_name, tool_description, max_tokens) -> LLMResult:
        t0 = time.perf_counter()
        try:
            r = self.client.chat.completions.create(
                model=self.model, max_completion_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": tool_name, "schema": schema, "strict": False}})
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"Lỗi gọi OpenAI API: {type(e).__name__}: {e}") from e
        ch = r.choices[0]
        if getattr(ch, "finish_reason", None) == "length":
            raise LLMError("Phản hồi bị cắt do hết max_tokens.")
        if getattr(ch.message, "refusal", None):
            raise LLMError(f"Model từ chối: {ch.message.refusal}")
        try:
            data = json.loads(ch.message.content or "")
        except json.JSONDecodeError as e:
            raise LLMError(f"Phản hồi không phải JSON hợp lệ: {e}") from e
        if not isinstance(data, dict):
            raise LLMError("Phản hồi JSON không phải đối tượng.")
        u = getattr(r, "usage", None)
        return LLMResult(data, self.name, getattr(r, "model", self.model),
                         getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None),
                         int((time.perf_counter() - t0) * 1000))


def make_provider(cfg: AIConfig, env=None) -> Optional[LLMProvider]:
    """None = chế độ ngoại tuyến (không có khoá API)."""
    import os
    e = os.environ if env is None else env
    name = cfg.resolve_provider_name(e)
    if name == "anthropic":
        if not e.get("ANTHROPIC_API_KEY"):
            raise LLMError("Thiếu biến môi trường ANTHROPIC_API_KEY.")
        return AnthropicProvider(e["ANTHROPIC_API_KEY"], cfg.model, cfg.timeout_s)
    if name == "openai":
        if not e.get("OPENAI_API_KEY"):
            raise LLMError("Thiếu biến môi trường OPENAI_API_KEY.")
        return OpenAIProvider(e["OPENAI_API_KEY"], cfg.model, cfg.timeout_s, base_url=cfg.openai_base_url)
    if name == "offline":
        return None
    raise LLMError(f"Provider không hỗ trợ: {name}")
