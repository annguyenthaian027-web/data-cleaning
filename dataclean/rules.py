"""Rule Engine: quy tắc làm sạch tuỳ chỉnh (JSON-serializable) + ghi audit log.

Mỗi rule = {action, column, params, enabled}. Rules chạy tuần tự theo thứ tự danh sách.
Action trả về (df_mới, danh_sách_thay_đổi[(row, column, old, new)]).
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from .detectors import (MISSING_TOKENS, duplicate_mask, is_valid_email, missing_mask,
                        normalize_phone, parse_date, to_num)

_NULL = "\x00NULL"


class RuleError(ValueError):
    pass


@dataclass
class Rule:
    action: str
    column: Union[str, list, None] = None       # None = tất cả cột
    params: dict = field(default_factory=dict)
    name: str = ""
    enabled: bool = True
    reason: str = ""                            # giải thích (dùng cho gợi ý)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Rule":
        allowed = {"action", "column", "params", "name", "enabled", "reason", "id"}
        return cls(**{k: v for k, v in d.items() if k in allowed})


# ---------------------------------------------------------------- helpers
def _diff(old: pd.Series, new: pd.Series, col: str) -> list[tuple]:
    o = old.astype(object).where(old.notna(), _NULL)
    n = new.astype(object).where(new.notna(), _NULL)
    idx = o.index[(o != n).to_numpy()]
    return [(i, col, None if o[i] == _NULL else o[i], None if n[i] == _NULL else n[i]) for i in idx]


def _cols(df: pd.DataFrame, rule: Rule) -> list[str]:
    c = rule.column
    cols = list(df.columns) if c is None else [c] if isinstance(c, str) else list(c)
    missing = [x for x in cols if x not in df.columns]
    if missing:
        raise RuleError(f"Không có cột: {missing}")
    return cols


def _fmt_num(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:.4f}".rstrip("0").rstrip(".")


def _map_col(df, col, fn):
    """Áp fn lên các ô không rỗng, trả về series mới (object)."""
    s = df[col]
    return s.map(lambda v: v if (v is None or (isinstance(v, float) and np.isnan(v))) else fn(v)).astype(object)


def _drop_rows(df, mask: pd.Series, reason: str):
    changes = [(i, None, reason, "(đã xoá dòng)") for i in df.index[mask]]
    return df.loc[~mask], changes


# ---------------------------------------------------------------- actions
ACTIONS: dict[str, Callable] = {}


def action(name):
    def deco(fn):
        ACTIONS[name] = fn
        return fn
    return deco


@action("trim_whitespace")
def _trim(df, rule):
    df, ch = df.copy(), []
    for c in _cols(df, rule):
        new = _map_col(df, c, lambda v: re.sub(r"\s+", " ", str(v)).strip())
        ch += _diff(df[c], new, c)
        df[c] = new
    return df, ch


@action("change_case")
def _case(df, rule):
    mode = rule.params.get("mode", "title")
    fns = {"lower": str.lower, "upper": str.upper, "title": str.title, "capitalize": str.capitalize}
    if mode not in fns:
        raise RuleError(f"mode phải thuộc {list(fns)}")
    df, ch = df.copy(), []
    for c in _cols(df, rule):
        new = _map_col(df, c, lambda v: fns[mode](str(v)))
        ch += _diff(df[c], new, c)
        df[c] = new
    return df, ch


@action("normalize_missing")
def _norm_missing(df, rule):
    """Đổi 'N/A', 'null', '-'... thành ô trống thật."""
    df, ch = df.copy(), []
    tokens = {t.lower() for t in rule.params.get("tokens", MISSING_TOKENS)}
    for c in _cols(df, rule):
        m = df[c].map(lambda v: isinstance(v, str) and v.strip().lower() in tokens).astype(bool)
        new = df[c].copy().astype(object)
        new[m] = np.nan
        ch += _diff(df[c], new, c)
        df[c] = new
    return df, ch


@action("fill_missing")
def _fill(df, rule):
    strategy = rule.params.get("strategy", "constant")
    df, ch = df.copy(), []
    for c in _cols(df, rule):
        m = missing_mask(df[c])
        if not m.any():
            continue
        valid = df[c][~m]
        if strategy in {"mean", "median"}:
            nums = to_num(valid).dropna()
            if nums.empty:
                continue
            val = _fmt_num(nums.mean() if strategy == "mean" else nums.median())
        elif strategy == "mode":
            if valid.empty:
                continue
            val = valid.mode().iloc[0]
        elif strategy == "constant":
            if "value" not in rule.params:
                raise RuleError("fill_missing/constant cần params.value")
            val = str(rule.params["value"])
        else:
            raise RuleError("strategy phải là mean|median|mode|constant")
        new = df[c].copy().astype(object)
        new[m] = val
        ch += _diff(df[c], new, c)
        df[c] = new
    return df, ch


@action("drop_missing_rows")
def _drop_missing(df, rule):
    cols = _cols(df, rule)
    mask = pd.concat([missing_mask(df[c]) for c in cols], axis=1).any(axis=1)
    return _drop_rows(df, mask, "Thiếu dữ liệu: " + ", ".join(cols))


@action("drop_duplicates")
def _drop_dups(df, rule):
    subset = None if rule.column is None else _cols(df, rule)
    mask = duplicate_mask(df, subset, rule.params.get("ignore_case", True))
    return _drop_rows(df, mask, "Dòng trùng lặp")


def _format_action(df, rule, convert, default_invalid="keep"):
    on_invalid = rule.params.get("on_invalid", default_invalid)
    if on_invalid not in {"keep", "null"}:
        raise RuleError("on_invalid phải là keep|null")
    df, ch = df.copy(), []
    for c in _cols(df, rule):
        def fn(v, _c=convert):
            r = _c(v)
            return r if r is not None else (v if on_invalid == "keep" else np.nan)
        new = _map_col(df, c, fn)
        ch += _diff(df[c], new, c)
        df[c] = new
    return df, ch


@action("normalize_date")
def _norm_date(df, rule):
    out_fmt = rule.params.get("output_format", "%Y-%m-%d")
    def conv(v):
        d = parse_date(v)
        return d.strftime(out_fmt) if d else None
    return _format_action(df, rule, conv)


@action("normalize_phone")
def _norm_phone(df, rule):
    return _format_action(df, rule, normalize_phone)


@action("validate_email")
def _val_email(df, rule):
    return _format_action(df, rule, lambda v: str(v).strip().lower() if is_valid_email(v) else None)


@action("convert_numeric")
def _num(df, rule):
    dec = rule.params.get("decimal", ".")
    df, ch = df.copy(), []
    on_invalid = rule.params.get("on_invalid", "keep")
    for c in _cols(df, rule):
        nums = to_num(df[c], dec)
        new = df[c].copy().astype(object)
        ok = nums.notna()
        new[ok] = nums[ok].map(_fmt_num)
        if on_invalid == "null":
            new[~ok & df[c].notna()] = np.nan
        ch += _diff(df[c], new, c)
        df[c] = new
    return df, ch


@action("handle_outliers")
def _outliers(df, rule):
    method = rule.params.get("method", "iqr")
    act = rule.params.get("action", "clip")
    k, z = rule.params.get("k", 1.5), rule.params.get("z", 3.0)
    if act not in {"clip", "null", "drop"}:
        raise RuleError("action phải là clip|null|drop")
    df, ch = df.copy(), []
    drop = pd.Series(False, index=df.index)
    for c in _cols(df, rule):
        nums = to_num(df[c])
        v = nums.dropna()
        if len(v) < 4:
            continue
        if method == "zscore":
            sd = v.std()
            if not sd:
                continue
            lo, hi = v.mean() - z * sd, v.mean() + z * sd
        else:
            q1, q3 = v.quantile(.25), v.quantile(.75)
            if q3 == q1:
                continue
            lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        bad = ((nums < lo) | (nums > hi)).fillna(False)
        if act == "drop":
            drop |= bad
            continue
        new = df[c].copy().astype(object)
        if act == "clip":
            new[bad] = nums[bad].clip(lo, hi).map(_fmt_num)
        else:
            new[bad] = np.nan
        ch += _diff(df[c], new, c)
        df[c] = new
    if act == "drop":
        return _drop_rows(df, drop, "Ngoại lai")
    return df, ch


@action("rename_column")
def _rename(df, rule):
    new_name = rule.params.get("new_name")
    if not isinstance(rule.column, str) or not new_name:
        raise RuleError("rename_column cần column (str) và params.new_name")
    _cols(df, rule)
    return df.rename(columns={rule.column: new_name}), [(None, rule.column, rule.column, new_name)]


# ---------------------------------------------------------------- engine
@dataclass
class ApplyResult:
    df: pd.DataFrame
    changes: list            # [(rule, [(row, col, old, new), ...]), ...]
    stats: dict


class RuleEngine:
    def __init__(self, rules: Optional[list] = None):
        self.rules: list[Rule] = [r if isinstance(r, Rule) else Rule.from_dict(r) for r in (rules or [])]
        for r in self.rules:
            self.validate(r)

    @staticmethod
    def validate(rule: Rule) -> None:
        if rule.action not in ACTIONS:
            raise RuleError(f"Action '{rule.action}' không tồn tại. Có: {sorted(ACTIONS)}")

    def add(self, rule: Rule) -> Rule:
        self.validate(rule)
        self.rules.append(rule)
        return rule

    def remove(self, rule_id: str) -> None:
        self.rules = [r for r in self.rules if r.id != rule_id]

    def toggle(self, rule_id: str, enabled: bool) -> None:
        for r in self.rules:
            if r.id == rule_id:
                r.enabled = enabled

    def to_json(self) -> str:
        return json.dumps([r.to_dict() for r in self.rules], ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "RuleEngine":
        return cls(json.loads(text))

    def apply(self, df: pd.DataFrame) -> ApplyResult:
        cur, all_changes = df.copy(), []
        for r in self.rules:
            if not r.enabled:
                continue
            cur, ch = ACTIONS[r.action](cur, r)
            all_changes.append((r, ch))
        n = sum(len(c) for _, c in all_changes)
        return ApplyResult(cur, all_changes, {
            "rules_run": len([r for r in self.rules if r.enabled]), "changes": n,
            "rows_before": len(df), "rows_after": len(cur), "rows_removed": len(df) - len(cur)})
