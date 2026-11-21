"""Chế độ ngoại tuyến: sinh phản hồi cùng định dạng với LLM từ phân tích rule-based của core.

Dùng khi chưa cấu hình khoá API hoặc khi API lỗi. Output đi qua cùng `RuleTranslator` như phản hồi của LLM.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

from dataclean.detectors import missing_mask

MONEY_HINTS = ("luong", "gia", "tien", "doanhthu", "salary", "price", "amount", "total", "cost", "revenue", "thanhtien")
NAME_HINTS = ("hoten", "ten", "name", "fullname", "hovaten")


def _flat(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn")
    return re.sub(r"[^0-9a-z]+", "", s.replace("đ", "d").replace("Đ", "D").lower())


def snake(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn")
    s = s.replace("đ", "d").replace("Đ", "D").lower()
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-z]+", "_", s)).strip("_") or "cot"


def _semantic(name: str, typ: str, unique_ratio: float) -> str:
    k = _flat(name)
    if typ == "email": return "email"
    if typ == "phone": return "phone"
    if typ == "date": return "date"
    if typ == "id": return "id"
    if typ == "numeric": return "money" if any(h in k for h in MONEY_HINTS) else "quantity"
    if any(k.endswith(h) or k == h for h in NAME_HINTS): return "person_name"
    return "category" if unique_ratio < 0.2 else "free_text"


_EXPLAIN = {
    "missing_value": ("Dữ liệu bị thiếu", "Có {n} ô trống hoặc ghi 'N/A', 'null', '-' ở {where}. Thiếu dữ liệu làm thống kê "
                      "(trung bình, đếm) sai lệch và có thể khiến một số chức năng không chạy."),
    "duplicate_row": ("Dòng trùng lặp", "Có {n} dòng giống hệt dòng khác. Nếu giữ nguyên, số liệu tổng hợp sẽ bị tính đôi."),
    "invalid_format": ("Sai định dạng", "Có {n} giá trị ở {where} không đúng định dạng mong đợi (ví dụ email thiếu '@', "
                       "ngày không tồn tại, số điện thoại quá ngắn). Cần người dùng kiểm tra lại giá trị gốc."),
    "nonstandard_format": ("Chưa đồng nhất", "Có {n} giá trị ở {where} đúng nội dung nhưng khác cách viết chuẩn (thừa khoảng trắng, "
                           "viết hoa/thường lẫn lộn, nhiều kiểu ngày/SĐT). Chuẩn hoá giúp so khớp và lọc chính xác hơn."),
    "outlier": ("Giá trị bất thường", "Có {n} giá trị ở {where} lớn/nhỏ bất thường so với phần còn lại. Có thể là nhập nhầm "
                "(thừa số 0) hoặc là dữ liệu thật — nên kiểm tra trước khi sửa hoặc xoá."),
}


def offline_response(df: pd.DataFrame, report: dict, tasks) -> dict:
    out: dict = {}
    types = {c["column"]: c["type"] for c in report["quality"]["per_column"]}
    if "schema" in tasks:
        cols = []
        for c in df.columns:
            valid = df[c][~missing_mask(df[c])]
            ur = (valid.nunique() / len(valid)) if len(valid) else 1.0
            sem = _semantic(c, types.get(c, "text"), ur)
            cols.append({"name": str(c), "semantic_type": sem, "description": f"Cột '{c}' — dữ liệu dạng {sem} "
                         f"(suy luận từ định dạng giá trị, không dùng AI).", "suggested_name": snake(c), "confidence": 0.5})
        out["schema"] = {"table_summary": f"Bảng {len(df):,} dòng × {len(df.columns)} cột (phân tích ngoại tuyến).",
                         "columns": cols}
    if "issues" in tasks:
        grp: dict = {}
        for i in report["issues"]:
            grp.setdefault((i["type"], i.get("column")), 0)
            grp[(i["type"], i.get("column"))] += 1
        findings = []
        for (t, col), n in sorted(grp.items(), key=lambda kv: -kv[1]):
            title, tpl = _EXPLAIN.get(t, (t, "Phát hiện {n} vấn đề ({where})."))
            ratio = n / max(report["rows"], 1)
            findings.append({"column": col, "issue_type": t, "severity": "high" if ratio > .2 else "medium" if ratio > .05 else "low",
                             "explanation": tpl.format(n=n, where=f"cột '{col}'" if col else "nhiều cột"),
                             "impact": title})
        tot = sum(report["issue_summary"].values())
        out["issues"] = {"summary": (f"Điểm chất lượng {report['quality']['overall']} ({report['quality']['grade']}); "
                                     f"phát hiện {tot} vấn đề trên {report['rows']:,} dòng." if tot else
                                     "Không phát hiện vấn đề nào."),
                         "findings": findings,
                         "next_steps": ["Xem trước và áp dụng các quy tắc rủi ro thấp trước.",
                                        "Tự kiểm tra các giá trị sai định dạng/ngoại lai vì máy không biết giá trị đúng."] if tot else []}
    if "rules" in tasks:
        out["rules"] = [{"action": r["action"], "column": r.get("column"), "params": r.get("params") or {},
                         "reason": r.get("reason", "")} for r in report["suggested_rules"]]
    return out
