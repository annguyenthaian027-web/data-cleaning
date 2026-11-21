"""Phân tích file: schema + lỗi + điểm chất lượng + gợi ý rule (rule-based).

`suggest_rules` trả về danh sách Rule kèm `reason` để lớp AI (Thành viên khác) có thể
diễn giải bằng ngôn ngữ dễ hiểu; người dùng xác nhận trước khi áp dụng.
"""
from __future__ import annotations

import re

import pandas as pd

from .detectors import Issue, detect_all, infer_column_types, missing_mask, to_num
from .quality import compute_quality
from .rules import Rule


def _has_messy_whitespace(df: pd.DataFrame) -> bool:
    for c in df.columns:
        for v in df[c]:
            if isinstance(v, str) and v != re.sub(r"\s+", " ", v).strip():
                return True
    return False


def suggest_rules(df: pd.DataFrame, types: dict, issues: list[Issue]) -> list[Rule]:
    by = {}
    for i in issues:
        by.setdefault((i.type, i.column), []).append(i)
    rules: list[Rule] = []

    if _has_messy_whitespace(df):
        rules.append(Rule("trim_whitespace", None, name="Xoá khoảng trắng thừa",
                          reason="Có ô chứa khoảng trắng đầu/cuối hoặc nhiều dấu cách liên tiếp."))
    if any(i.value is not None for i in issues if i.type == "missing_value"):
        rules.append(Rule("normalize_missing", None, name="Chuẩn hoá giá trị rỗng (N/A, null, -)",
                          reason="Có ô ghi 'N/A', 'null', '-'... thực chất là dữ liệu thiếu."))
    for col, t in types.items():
        fmt_issues = by.get(("invalid_format", col), []) + by.get(("nonstandard_format", col), [])
        if t == "date" and fmt_issues:
            rules.append(Rule("normalize_date", col, {"output_format": "%Y-%m-%d"},
                              name=f"Chuẩn hoá ngày: {col}",
                              reason=f"{len(fmt_issues)} ô ngày sai/khác định dạng."))
        elif t == "phone" and fmt_issues:
            rules.append(Rule("normalize_phone", col, name=f"Chuẩn hoá SĐT: {col}",
                              reason=f"{len(fmt_issues)} số điện thoại sai/khác định dạng."))
        elif t == "email" and fmt_issues:
            rules.append(Rule("validate_email", col, {"on_invalid": "null"}, name=f"Kiểm tra email: {col}",
                              reason=f"{len(fmt_issues)} email lỗi hoặc có chữ hoa/khoảng trắng."))
        elif t == "numeric":
            if any(re.search(r"[,\s]", str(v)) for v in df[col].dropna()):
                rules.append(Rule("convert_numeric", col, name=f"Chuyển số: {col}",
                                  reason="Có số chứa dấu phân cách nghìn/khoảng trắng."))
    for col, t in types.items():          # ngoại lai xử lý TRƯỚC khi điền thiếu (điền median làm hẹp IQR)
        if t == "numeric" and by.get(("outlier", col)):
            rules.append(Rule("handle_outliers", col, {"method": "iqr", "action": "clip"},
                              name=f"Giới hạn ngoại lai: {col}",
                              reason=f"{len(by[('outlier', col)])} giá trị ngoại lai (IQR)."))
    for col, t in types.items():
        if t == "numeric" and by.get(("missing_value", col)):
            rules.append(Rule("fill_missing", col, {"strategy": "median"}, name=f"Điền thiếu bằng median: {col}",
                              reason=f"{len(by[('missing_value', col)])} ô số bị thiếu."))
    if any(i.type == "duplicate_row" for i in issues):
        n = sum(1 for i in issues if i.type == "duplicate_row")
        rules.append(Rule("drop_duplicates", None, name="Xoá dòng trùng lặp", reason=f"{n} dòng trùng."))
    return rules


def analyze(df: pd.DataFrame) -> dict:
    """Điểm vào cho API: trả về dict JSON-serializable."""
    types = infer_column_types(df)
    issues = detect_all(df, types)
    summary: dict[str, int] = {}
    for i in issues:
        summary[i.type] = summary.get(i.type, 0) + 1
    return {
        "rows": len(df), "columns": list(df.columns), "types": types,
        "issue_summary": summary,
        "issues": [i.to_dict() for i in issues],
        "quality": compute_quality(df, types, issues),
        "suggested_rules": [r.to_dict() for r in suggest_rules(df, types, issues)],
    }
