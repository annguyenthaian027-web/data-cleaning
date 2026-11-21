"""Xuất dữ liệu sạch (CSV / Excel) kèm audit log & báo cáo chất lượng."""
from __future__ import annotations

import io

import pandas as pd

from .quality import quality_table


def export_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")        # BOM để Excel đọc đúng tiếng Việt


def export_excel_bytes(df: pd.DataFrame, audit_df: pd.DataFrame | None = None,
                       quality: dict | None = None) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="Clean data", index=False)
        if quality is not None:
            quality_table(quality).to_excel(xw, sheet_name="Quality report", index=False)
        if audit_df is not None:
            audit_df.to_excel(xw, sheet_name="Audit log", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 50)
            ws.freeze_panes = "A2"
    return buf.getvalue()


def export_session(session, fmt: str = "xlsx", include_report: bool = True) -> tuple[bytes, str, str]:
    """Trả về (bytes, filename, mime) để FastAPI/Streamlit trả về cho người dùng."""
    from .quality import compute_quality
    stem = session.filename.rsplit(".", 1)[0]
    if fmt == "csv":
        return export_csv_bytes(session.current), f"{stem}_clean.csv", "text/csv"
    data = export_excel_bytes(session.current, session.audit.to_dataframe() if include_report else None,
                              compute_quality(session.current) if include_report else None)
    return data, f"{stem}_clean.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
