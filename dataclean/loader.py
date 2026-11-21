"""Đọc file upload (CSV / Excel) thành DataFrame toàn kiểu chuỗi.

Toàn bộ dữ liệu được đọc dạng str để KHÔNG làm mất thông tin gốc
(số 0 đầu của số điện thoại, định dạng ngày...). Việc ép kiểu do Rule Engine đảm nhiệm.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Union

import pandas as pd

SUPPORTED_EXT = {".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".xls"}
CSV_ENCODINGS = ["utf-8-sig", "utf-8", "cp1258", "cp1252", "latin-1"]
MAX_ROWS_DEFAULT = 500_000


class FileLoadError(Exception):
    """Lỗi người dùng có thể đọc được (file sai định dạng, rỗng, quá lớn...)."""


Source = Union[str, Path, bytes, BinaryIO]


def _to_buffer(source: Source) -> Union[str, Path, io.BytesIO]:
    if isinstance(source, (bytes, bytearray)):
        return io.BytesIO(source)
    if hasattr(source, "read"):
        return io.BytesIO(source.read())
    return source


def _read_csv(buf, sep) -> pd.DataFrame:
    raw = buf.getvalue() if isinstance(buf, io.BytesIO) else Path(buf).read_bytes()
    last_err = None
    for enc in CSV_ENCODINGS:
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
        return pd.read_csv(
            io.StringIO(text), sep=sep, engine="python", dtype=str,
            keep_default_na=False, na_values=[""], skip_blank_lines=True,
        )
    raise FileLoadError(f"Không đọc được encoding của file CSV: {last_err}")


def _read_excel(buf, sheet_name) -> pd.DataFrame:
    try:
        return pd.read_excel(buf, sheet_name=sheet_name, dtype=str,
                             keep_default_na=False, na_values=[""])
    except ValueError as e:
        raise FileLoadError(f"Không đọc được file Excel: {e}") from e


def load_dataframe(source: Source, filename: str | None = None,
                   sheet_name: Union[int, str] = 0,
                   max_rows: int = MAX_ROWS_DEFAULT) -> pd.DataFrame:
    """Đọc CSV/Excel -> DataFrame (index 0..n-1, mọi ô là str hoặc NaN)."""
    name = filename or (str(source) if isinstance(source, (str, Path)) else "")
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise FileLoadError(f"Định dạng '{ext or '?'}' chưa hỗ trợ. Chỉ nhận: {sorted(SUPPORTED_EXT)}")

    buf = _to_buffer(source)
    if ext in {".csv", ".txt"}:
        df = _read_csv(buf, sep=None)      # tự đoán dấu phân cách , ; |
    elif ext == ".tsv":
        df = _read_csv(buf, sep="\t")
    else:
        df = _read_excel(buf, sheet_name)

    if df.empty and len(df.columns) == 0:
        raise FileLoadError("File rỗng.")
    if len(df) > max_rows:
        raise FileLoadError(f"File có {len(df):,} dòng, vượt giới hạn {max_rows:,}.")

    df = _tidy(df)
    return df


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    # chuẩn hoá tên cột: strip, đặt tên cho cột trống, tránh trùng tên
    seen: dict[str, int] = {}
    cols = []
    for i, c in enumerate(df.columns):
        c = str(c).strip()
        if not c or c.lower().startswith("unnamed:"):
            c = f"column_{i + 1}"
        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
        cols.append(c)
    df.columns = cols
    df = df.astype(object).where(df.notna(), None)
    # Excel trả datetime dạng "2024-01-05 00:00:00" -> bỏ phần giờ rỗng
    df = df.apply(lambda s: s.map(
        lambda v: v[:-9] if isinstance(v, str) and v.endswith(" 00:00:00") else v))
    df = df.dropna(how="all").reset_index(drop=True)   # bỏ dòng trống hoàn toàn
    return df.astype(object).where(df.notna(), float("nan"))


def list_sheets(source: Source) -> list[str]:
    """Liệt kê sheet của file Excel (để UI cho người dùng chọn)."""
    return pd.ExcelFile(_to_buffer(source)).sheet_names
