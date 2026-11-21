"""Các thuật toán tự động phát hiện lỗi dữ liệu.

Lỗi: trùng lặp, thiếu dữ liệu, ngoại lai (IQR/z-score), sai định dạng (ngày, SĐT VN, email).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}$")
PHONE_VN_RE = re.compile(r"^0[35789]\d{8}$")            # dạng chuẩn 10 số
MISSING_TOKENS = {"", "na", "n/a", "null", "none", "nan", "-", "--", "?", "không có", "khong co"}
DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y/%m/%d",
    "%d/%m/%y", "%d-%m-%y", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S",
]
ID_TOKENS = {"id", "ma", "code", "stt", "no"}
HINTS = {
    "email": {"email", "mail"},
    "phone": {"phone", "sdt", "tel", "mobile", "dienthoai", "didong"},
    "date": {"date", "ngay", "dob", "birth", "sinh", "created", "updated"},
}


# ---------------------------------------------------------------- helpers
@dataclass
class Issue:
    type: str                 # missing_value | duplicate_row | outlier | invalid_format | nonstandard_format
    severity: str             # error | warning
    column: Optional[str]
    row: Optional[int]        # index dòng trong DataFrame (0-based); Excel row = row + 2
    value: Optional[str]
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def _unaccent(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).lower().replace("đ", "d"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _is_missing_value(v) -> bool:
    return v is None or (isinstance(v, float) and v != v) or str(v).strip().lower() in MISSING_TOKENS


def missing_mask(s: pd.Series) -> pd.Series:
    return pd.Series([_is_missing_value(v) for v in s], index=s.index, dtype=bool)


def parse_date(v) -> Optional[datetime]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    t = str(v).strip()
    for f in DATE_FORMATS:
        try:
            return datetime.strptime(t, f)
        except ValueError:
            continue
    return None


def date_pattern(v) -> Optional[str]:
    t = str(v).strip()
    for f in DATE_FORMATS:
        try:
            datetime.strptime(t, f)
            return f
        except ValueError:
            continue
    return None


def normalize_phone(v) -> Optional[str]:
    """Trả về SĐT VN dạng 0xxxxxxxxx hoặc None nếu không hợp lệ."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    d = re.sub(r"[\s.\-()]", "", str(v))
    if d.startswith("+84"):
        d = "0" + d[3:]
    elif d.startswith("84") and len(d) == 11:
        d = "0" + d[2:]
    elif len(d) == 9 and d[0] in "35789":          # Excel làm mất số 0 đầu
        d = "0" + d
    return d if PHONE_VN_RE.match(d) else None


def is_valid_email(v) -> bool:
    return bool(EMAIL_RE.match(str(v).strip()))


_NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _parse_num(v, decimal: str = ".") -> float:
    if v is None or (isinstance(v, float) and v != v):
        return np.nan
    t = re.sub(r"\s", "", str(v))
    t = t.replace(".", "").replace(",", ".") if decimal == "," else t.replace(",", "")
    return float(t) if _NUM_RE.match(t) else np.nan


def to_num(s: pd.Series, decimal: str = ".") -> pd.Series:
    return pd.Series([_parse_num(v, decimal) for v in s], index=s.index, dtype="float64")


def duplicate_mask(df: pd.DataFrame, subset=None, ignore_case: bool = True) -> pd.Series:
    """True cho các dòng trùng (giữ lại dòng xuất hiện đầu tiên)."""
    cols = list(subset) if subset else list(df.columns)
    if not cols:
        return pd.Series(False, index=df.index)

    def norm(v):
        if v is None or (isinstance(v, float) and v != v):
            return ""
        t = str(v).strip()
        return re.sub(r"\s+", " ", t).lower() if ignore_case else t

    seen, out = set(), []
    for key in zip(*[[norm(v) for v in df[c]] for c in cols]):
        out.append(key in seen)
        seen.add(key)
    return pd.Series(out, index=df.index, dtype=bool)


# ---------------------------------------------------------------- type inference
def infer_column_types(df: pd.DataFrame, threshold: float = 0.7) -> dict[str, str]:
    """Phân tích schema: email | phone | date | numeric | id | text."""
    types: dict[str, str] = {}
    for col in df.columns:
        vals = df[col][~missing_mask(df[col])]
        tokens = set(re.split(r"[^a-z0-9]+", _unaccent(col))) | {_unaccent(col).replace(" ", "")}
        if tokens & ID_TOKENS and not tokens & (HINTS["phone"] | HINTS["email"]):
            types[col] = "id"
            continue
        if len(vals) == 0:
            types[col] = "text"
            continue

        r_email = np.mean([is_valid_email(v) for v in vals])
        r_phone = np.mean([normalize_phone(v) is not None for v in vals])
        r_date = np.mean([parse_date(v) is not None for v in vals])
        r_num = to_num(vals).notna().mean()

        def hinted(kind):   # tên cột gợi ý -> chấp nhận ngưỡng thấp để bắt được giá trị lỗi
            return bool(tokens & HINTS[kind] or any(h in _unaccent(col).replace(" ", "") for h in HINTS[kind]))

        if r_email >= threshold or (hinted("email") and r_email >= 0.3):
            types[col] = "email"
        elif r_phone >= threshold or (hinted("phone") and r_phone >= 0.3):
            types[col] = "phone"
        elif r_date >= threshold or (hinted("date") and r_date >= 0.3):
            types[col] = "date"
        elif r_num >= 0.8:
            types[col] = "numeric"
        else:
            types[col] = "text"
    return types


# ---------------------------------------------------------------- detectors
def detect_missing(df: pd.DataFrame) -> list[Issue]:
    out = []
    for col in df.columns:
        m = missing_mask(df[col])
        for i in df.index[m]:
            raw = df.at[i, col]
            out.append(Issue("missing_value", "warning", col, int(i),
                             None if pd.isna(raw) else str(raw), f"Thiếu dữ liệu ở cột '{col}'"))
    return out


def detect_duplicates(df: pd.DataFrame, subset=None) -> list[Issue]:
    m = duplicate_mask(df, subset)
    return [Issue("duplicate_row", "error", None, int(i), None,
                  "Dòng trùng lặp với một dòng phía trên") for i in df.index[m]]


def detect_outliers(df: pd.DataFrame, types: dict[str, str], method: str = "iqr",
                    k: float = 1.5, z: float = 3.0) -> list[Issue]:
    out = []
    for col, t in types.items():
        if t != "numeric":
            continue
        nums = to_num(df[col])
        v = nums.dropna()
        if len(v) < 4:
            continue
        if method == "zscore":
            sd = v.std()
            if not sd:
                continue
            bad = ((nums - v.mean()).abs() / sd) > z
            lo, hi = v.mean() - z * sd, v.mean() + z * sd
        else:
            q1, q3 = v.quantile(0.25), v.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo, hi = q1 - k * iqr, q3 + k * iqr
            bad = (nums < lo) | (nums > hi)
        for i in nums.index[bad.fillna(False)]:
            out.append(Issue("outlier", "warning", col, int(i), str(df.at[i, col]),
                             f"Ngoại lai: ngoài khoảng [{lo:.2f}, {hi:.2f}]"))
    return out


def detect_format_errors(df: pd.DataFrame, types: dict[str, str]) -> list[Issue]:
    out = []
    for col, t in types.items():
        if t not in {"email", "phone", "date"}:
            continue
        s = df[col]
        ok = ~missing_mask(s)
        if t == "date":
            pats = {i: date_pattern(s[i]) for i in s.index[ok]}
            counts = pd.Series([p for p in pats.values() if p]).value_counts()
            dominant = counts.index[0] if len(counts) else None
        for i in s.index[ok]:
            v = str(s[i])
            if t == "email":
                if not is_valid_email(v):
                    out.append(Issue("invalid_format", "error", col, int(i), v, "Email không hợp lệ"))
                elif v != v.strip().lower():
                    out.append(Issue("nonstandard_format", "warning", col, int(i), v, "Email có khoảng trắng/chữ hoa"))
            elif t == "phone":
                n = normalize_phone(v)
                if n is None:
                    out.append(Issue("invalid_format", "error", col, int(i), v, "Số điện thoại không hợp lệ"))
                elif n != v:
                    out.append(Issue("nonstandard_format", "warning", col, int(i), v,
                                     f"SĐT chưa chuẩn hoá (chuẩn: {n})"))
            else:
                p = pats[i]
                if p is None:
                    out.append(Issue("invalid_format", "error", col, int(i), v, "Ngày không hợp lệ"))
                elif p != dominant:
                    out.append(Issue("nonstandard_format", "warning", col, int(i), v,
                                     "Định dạng ngày khác với định dạng phổ biến của cột"))
    return out


def detect_all(df: pd.DataFrame, types: dict[str, str] | None = None,
               outlier_method: str = "iqr") -> list[Issue]:
    types = types or infer_column_types(df)
    return (detect_missing(df) + detect_duplicates(df)
            + detect_format_errors(df, types) + detect_outliers(df, types, outlier_method))
