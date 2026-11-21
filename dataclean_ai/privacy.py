"""Tạo "hồ sơ schema" gửi cho LLM — nguyên tắc: CHỈ gửi thứ cần thiết, và che dữ liệu nhạy cảm.

Chế độ (AIConfig.sample_mode):
  none    : tên cột, kiểu suy luận, số ô thiếu/duy nhất, số lỗi theo loại. Không có giá trị nào.
  pattern : (mặc định) thêm "chữ ký định dạng" đã che: 'Nguyễn Văn A' -> 'Aaaaaa Aaa A', '0912345678' -> '9{10}'.
  raw     : thêm vài giá trị thật + min/max của cột số (chỉ bật khi người dùng đồng ý rõ ràng).
Không bao giờ gửi nguyên cả dòng hay toàn bộ bảng.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

import pandas as pd

from dataclean.detectors import missing_mask, to_num

_CTRL = re.compile(r"[\x00-\x1f\x7f]")


def clean_name(name, limit: int = 120) -> str:
    """Tên cột là dữ liệu KHÔNG tin cậy (có thể chứa lệnh) -> bỏ ký tự điều khiển, giới hạn độ dài."""
    return _CTRL.sub(" ", str(name)).strip()[:limit]


def mask_value(v, max_len: int = 30) -> str:
    """Giữ 'hình dạng' giá trị: chữ hoa->A, thường->a, số->9, ký tự khác giữ nguyên; nén chuỗi lặp dài."""
    prev, run = None, 0
    s = _CTRL.sub(" ", str(v).strip())
    parts = []
    for ch in s:
        cls = "9" if ch.isdigit() else ("A" if ch.isalpha() and ch.isupper() else ("a" if ch.isalpha() else ch))
        if cls == prev:
            run += 1
        else:
            if prev is not None:
                parts.append((prev, run))
            prev, run = cls, 1
    if prev is not None:
        parts.append((prev, run))
    res = "".join((c * n if n <= 3 or not c.isalnum() else f"{c}{{{n}}}") for c, n in parts)
    return res[:max_len]


def build_profile(df: pd.DataFrame, report: dict, sample_mode: str = "pattern", max_columns: int = 60,
                  pattern_sample: int = 2000) -> dict:
    types = {c["column"]: c["type"] for c in report["quality"]["per_column"]}
    by_col: dict = {}
    for i in report["issues"]:
        if i.get("column") is not None:
            by_col.setdefault(i["column"], Counter())[i["type"]] += 1
    cols = list(df.columns)[:max_columns]
    prof_cols = []
    for c in cols:
        s = df[c]
        m = missing_mask(s)
        valid = s[~m]
        entry = {"name": clean_name(c), "inferred_type": types.get(c, "text"),
                 "missing": int(m.sum()), "unique": int(valid.nunique()),
                 "issue_counts": dict(by_col.get(c, {}))}
        if sample_mode in ("pattern", "raw") and len(valid):
            smp = valid.head(pattern_sample) if len(valid) <= pattern_sample else valid.sample(pattern_sample, random_state=0)
            cnt = Counter(mask_value(v) for v in smp)
            entry["patterns"] = [{"pattern": p, "count": n} for p, n in cnt.most_common(5)]
            entry["patterns_sampled_rows"] = int(len(smp))
        if sample_mode == "raw" and len(valid):
            entry["samples"] = [str(v)[:40] for v in valid.drop_duplicates().head(5)]
            if entry["inferred_type"] == "numeric":
                nums = to_num(valid, "auto").dropna()
                if len(nums):
                    entry["numeric_range"] = {"min": float(nums.min()), "max": float(nums.max())}
        prof_cols.append(entry)
    return {"row_count": int(len(df)), "column_count": int(len(df.columns)),
            "columns_truncated": len(df.columns) > max_columns,
            "issue_summary": dict(report["issue_summary"]),
            "quality": {"overall": report["quality"]["overall"], "dimensions": report["quality"]["dimensions"]},
            "sample_mode": sample_mode, "columns": prof_cols}
