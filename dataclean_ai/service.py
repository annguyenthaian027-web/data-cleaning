"""AIAdvisor: điều phối  hồ sơ (che dữ liệu) → LLM → phân tích kết quả → bộ kiểm duyệt → Rule."""
from __future__ import annotations

import copy
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from dataclean import Rule, analyze
from dataclean.detectors import infer_column_types

from . import prompts
from .config import AIConfig
from .fallback import offline_response
from .privacy import build_profile
from .providers import LLMError, LLMProvider, make_provider
from .translator import RuleTranslator, TranslationResult, resolve_column

_UNSET = object()


@dataclass
class ColumnInsight:
    name: str
    semantic_type: str
    description: str
    suggested_name: str
    confidence: float


@dataclass
class Finding:
    column: Optional[str]
    issue_type: str
    explanation: str
    impact: str
    severity: str


@dataclass
class AIReview:
    tasks: tuple
    provider: str
    model: str
    degraded: bool = False                 # True = dùng chế độ ngoại tuyến (không có/lỗi LLM)
    error: Optional[str] = None
    cache_hit: bool = False
    table_summary: str = ""
    columns: list = field(default_factory=list)
    issues_summary: str = ""
    findings: list = field(default_factory=list)
    next_steps: list = field(default_factory=list)
    translation: Optional[TranslationResult] = None
    profile: dict = field(default_factory=dict)
    profile_hash: str = ""
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def rules(self) -> list:
        return self.translation.rules if self.translation else []

    @property
    def proposals(self) -> list:
        return self.translation.proposals if self.translation else []

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("tasks", "provider", "model", "degraded", "error", "cache_hit", "table_summary",
                                           "issues_summary", "next_steps", "profile_hash", "usage")}
        d["tasks"] = list(self.tasks)
        d["columns"] = [vars(c) for c in self.columns]
        d["findings"] = [vars(f) for f in self.findings]
        d["translation"] = self.translation.to_dict() if self.translation else None
        return d


def _f(v, lo=0.0, hi=1.0, default=0.5) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return default


def _s(v, n=600) -> str:
    return "" if v is None else str(v)[:n]


class AIAdvisor:
    def __init__(self, config: Optional[AIConfig] = None, provider=_UNSET, cache_size: int = 32):
        self.config = config or AIConfig.from_env()
        self.init_error: Optional[str] = None
        if provider is _UNSET:
            try:
                provider = make_provider(self.config)
            except LLMError as e:
                provider, self.init_error = None, str(e)
        self.provider: Optional[LLMProvider] = provider
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size

    @property
    def mode(self) -> str:
        return f"{self.provider.name}:{self.provider.model}" if self.provider else "offline"

    # ---------------------------------------------------------------- API công khai
    def review(self, df: pd.DataFrame, report: Optional[dict] = None, tasks=prompts.TASKS) -> AIReview:
        tasks = tuple(t for t in prompts.TASKS if t in tasks)
        if not tasks:
            raise ValueError(f"tasks phải thuộc {prompts.TASKS}")
        report = report or analyze(df)
        cfg = self.config
        profile = build_profile(df, report, cfg.sample_mode, cfg.max_columns)
        phash = hashlib.sha256(json.dumps([profile, tasks, self.mode], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        rv = AIReview(tasks=tasks, provider=self.provider.name if self.provider else "offline",
                      model=self.provider.model if self.provider else "rule-based", profile=profile, profile_hash=phash)

        if phash in self._cache:
            self._cache.move_to_end(phash)
            data, usage, degraded, err = copy.deepcopy(self._cache[phash])
            rv.cache_hit = True
        else:
            data, usage, degraded, err = self._obtain(df, report, profile, tasks)
            self._cache[phash] = copy.deepcopy((data, usage, degraded, err))
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        rv.raw, rv.usage, rv.degraded, rv.error = data, ({} if rv.cache_hit else usage), degraded, err
        if rv.degraded:
            rv.provider, rv.model = "offline", "rule-based"
        self._parse(rv, data, df)
        return rv

    def analyze_schema(self, df, report=None) -> AIReview:
        return self.review(df, report, ("schema",))

    def explain_issues(self, df, report=None) -> AIReview:
        return self.review(df, report, ("issues",))

    def suggest_rules(self, df, report=None) -> AIReview:
        return self.review(df, report, ("rules",))

    # ---------------------------------------------------------------- nội bộ
    def _obtain(self, df, report, profile, tasks):
        if self.provider is None:
            return offline_response(df, report, tasks), {}, True, self.init_error
        schema = prompts.build_schema(tasks)
        last: Optional[Exception] = None
        for _ in range(max(1, self.config.max_attempts)):
            try:
                r = self.provider.generate_json(
                    system=prompts.system_prompt(), user=prompts.user_prompt(profile, tasks), schema=schema,
                    tool_name=prompts.TOOL_NAME, tool_description=prompts.TOOL_DESCRIPTION,
                    max_tokens=self.config.max_tokens)
                missing = [t for t in tasks if (t if t != "issues" else "issues") not in r.data]
                if missing:
                    raise LLMError(f"Phản hồi thiếu phần: {missing}")
                return r.data, {"input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                                "latency_ms": r.latency_ms}, False, None
            except LLMError as e:
                last = e
        if self.config.fallback_on_error:
            return offline_response(df, report, tasks), {}, True, str(last)
        raise last  # type: ignore[misc]

    def _parse(self, rv: AIReview, data: dict, df: pd.DataFrame) -> None:
        cols = list(df.columns)
        raw_rules: list = []
        if "schema" in rv.tasks:
            sc = data.get("schema") if isinstance(data.get("schema"), dict) else {}
            rv.table_summary = _s(sc.get("table_summary"))
            for c in (sc.get("columns") or []) if isinstance(sc.get("columns"), list) else []:
                if not isinstance(c, dict):
                    continue
                real, _ = resolve_column(c.get("name"), cols)
                if real is None:
                    continue                                   # AI nhắc tới cột không tồn tại -> bỏ
                st = c.get("semantic_type")
                ci = ColumnInsight(real, st if st in prompts.SEMANTIC_TYPES else "other", _s(c.get("description"), 300),
                                   _s(c.get("suggested_name"), 100) or real, _f(c.get("confidence")))
                rv.columns.append(ci)
                if ci.suggested_name and ci.suggested_name != real and ci.confidence >= 0.6:
                    raw_rules.append({"action": "rename_column", "column": real, "params": {"new_name": ci.suggested_name},
                                      "reason": f"Tên chuẩn hoá theo ý nghĩa cột ({ci.semantic_type}): {ci.description}"[:300],
                                      "name": f"Đổi tên cột: {real} → {ci.suggested_name}"})
        if "issues" in rv.tasks:
            iss = data.get("issues") if isinstance(data.get("issues"), dict) else {}
            rv.issues_summary = _s(iss.get("summary"))
            for f in (iss.get("findings") or []) if isinstance(iss.get("findings"), list) else []:
                if not isinstance(f, dict):
                    continue
                col = f.get("column")
                real, _ = resolve_column(col, cols) if col else (None, None)
                sev = f.get("severity")
                rv.findings.append(Finding(real if col else None, _s(f.get("issue_type"), 60), _s(f.get("explanation")),
                                           _s(f.get("impact"), 300), sev if sev in ("low", "medium", "high") else "low"))
            ns = iss.get("next_steps")
            rv.next_steps = [_s(x, 300) for x in ns][:8] if isinstance(ns, list) else []
        if "rules" in rv.tasks and isinstance(data.get("rules"), list):
            raw_rules = [r for r in data["rules"]] + raw_rules
        if raw_rules:
            rv.translation = RuleTranslator(df, infer_column_types(df), self.config.dry_run_rows).translate(raw_rules)
