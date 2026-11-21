"""Cấu hình tích hợp AI (đọc từ biến môi trường; KHÔNG bao giờ ghi khoá API vào code/DB)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

# Tên model chỉ là mặc định: đặt DATACLEAN_LLM_MODEL theo model tài khoản của bạn được dùng.
DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4o-mini"}
SAMPLE_MODES = ("none", "pattern", "raw")


@dataclass(frozen=True)
class AIConfig:
    provider: str = "auto"              # auto | anthropic | openai | offline
    model: Optional[str] = None
    sample_mode: str = "pattern"        # none: chỉ tên cột+thống kê | pattern: thêm "hình dạng" giá trị đã che | raw: giá trị thật
    max_columns: int = 60
    max_tokens: int = 4000
    timeout_s: float = 60.0
    max_attempts: int = 2
    fallback_on_error: bool = True      # lỗi API -> dùng gợi ý rule-based thay vì làm sập luồng
    openai_base_url: Optional[str] = None
    dry_run_rows: int = 5000

    def __post_init__(self):
        if self.sample_mode not in SAMPLE_MODES:
            raise ValueError(f"sample_mode phải thuộc {SAMPLE_MODES}")

    @classmethod
    def from_env(cls, env=None) -> "AIConfig":
        e = os.environ if env is None else env
        return cls(provider=e.get("DATACLEAN_LLM_PROVIDER", "auto").lower(),
                   model=e.get("DATACLEAN_LLM_MODEL") or None,
                   sample_mode=e.get("DATACLEAN_AI_SAMPLE_MODE", "pattern").lower(),
                   openai_base_url=e.get("OPENAI_BASE_URL") or None)

    def resolve_provider_name(self, env=None) -> str:
        e = os.environ if env is None else env
        if self.provider != "auto":
            return self.provider
        if e.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        if e.get("OPENAI_API_KEY"):
            return "openai"
        return "offline"
