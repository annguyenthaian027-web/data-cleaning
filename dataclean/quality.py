"""Data quality score (0-100) = trung bình có trọng số của 4 chiều chất lượng."""
from __future__ import annotations

import pandas as pd

from .detectors import Issue, detect_all, infer_column_types, missing_mask

WEIGHTS = {"completeness": 0.30, "uniqueness": 0.20, "validity": 0.30, "consistency": 0.20}


def _grade(score: float) -> str:
    return "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D" if score >= 60 else "F"


def compute_quality(df: pd.DataFrame, types: dict | None = None, issues: list[Issue] | None = None) -> dict:
    if df.empty:
        return {"overall": 0.0, "grade": "F", "dimensions": {k: 0.0 for k in WEIGHTS}, "counts": {}, "per_column": []}
    types = types or infer_column_types(df)
    issues = detect_all(df, types) if issues is None else issues
    n_rows, n_cols = df.shape

    cnt = {t: 0 for t in ["missing_value", "duplicate_row", "invalid_format", "nonstandard_format", "outlier"]}
    for i in issues:
        cnt[i.type] = cnt.get(i.type, 0) + 1

    fmt_cols = [c for c, t in types.items() if t in {"email", "phone", "date"}]
    num_cols = [c for c, t in types.items() if t == "numeric"]
    nonmissing = {c: int((~missing_mask(df[c])).sum()) for c in df.columns}
    checked_fmt = sum(nonmissing[c] for c in fmt_cols)
    checked_cons = checked_fmt + sum(nonmissing[c] for c in num_cols)

    ratio = lambda bad, total: 1.0 if total == 0 else max(0.0, 1 - bad / total)
    dims = {
        "completeness": ratio(cnt["missing_value"], n_rows * n_cols),
        "uniqueness": ratio(cnt["duplicate_row"], n_rows),
        "validity": ratio(cnt["invalid_format"], checked_fmt),
        "consistency": ratio(cnt["nonstandard_format"] + cnt["outlier"], checked_cons),
    }
    overall = sum(dims[k] * w for k, w in WEIGHTS.items()) * 100
    per_col = []
    for c in df.columns:
        n_issue = sum(1 for i in issues if i.column == c)
        per_col.append({"column": c, "type": types[c], "missing": n_rows - nonmissing[c], "issues": n_issue})
    return {"overall": round(overall, 1), "grade": _grade(overall),
            "dimensions": {k: round(v * 100, 1) for k, v in dims.items()},
            "counts": cnt, "rows": n_rows, "columns": n_cols, "per_column": per_col}


def quality_table(q: dict) -> pd.DataFrame:
    rows = [("Overall score", q["overall"]), ("Grade", q["grade"])]
    rows += [(f"Dimension: {k}", v) for k, v in q["dimensions"].items()]
    rows += [(f"Issues: {k}", v) for k, v in q.get("counts", {}).items()]
    return pd.DataFrame(rows, columns=["Metric", "Value"])
