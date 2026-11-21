"""Chạy thử toàn bộ pipeline: python demo.py"""
import io, json
import pandas as pd
from dataclean import *

csv = """Mã KH,Họ tên,Email,Số điện thoại,Ngày sinh,Lương
KH01,  nguyễn văn a ,A@Gmail.com,0912345678,05/01/1990,15000000
KH02,Trần Thị B,b@@gmail,+84 912 345 679,1992-03-15,"16,000,000"
KH03,Lê Văn C,c@x.vn,912345680,31/02/1991,14000000
KH04,Phạm D,N/A,0987.654.321,20-12-1989,15500000
KH05,Hoàng E,e@x.vn,0901234567,10/10/1995,15200000
KH05,hoàng e,e@x.vn,0901234567,10/10/1995,15200000
KH06,Võ F,f@x.vn,0933333333,01/01/1993,
KH07,Đỗ G,g@x.vn,0944444444,02/02/1994,900000000
KH08,Bùi H,h@x.vn,0955555555,03/03/1994,14800000
"""
df = load_dataframe(csv.encode("utf-8"), "khach_hang.csv")
rep = analyze(df)
print("Kiểu cột:", rep["types"])
print("Lỗi:", rep["issue_summary"])
print("Điểm trước:", rep["quality"]["overall"], rep["quality"]["grade"], rep["quality"]["dimensions"])
print("Gợi ý:"); [print(" -", r["name"], "|", r["reason"]) for r in rep["suggested_rules"]]

s = CleaningSession(df, "khach_hang.csv")
rules = [Rule.from_dict(r) for r in rep["suggested_rules"]]
rules.append(Rule("change_case", "Họ tên", {"mode": "title"}, name="Viết hoa tên"))
rules.insert(2, rules.pop())  # đặt sau trim/normalize_missing
pv = s.preview(rules)
print("Preview:", pv["stats"], pv["quality_before"], "->", pv["quality_after"])
s.apply(rules, "Làm sạch lần 1")
print(s.current.to_string())
print("Điểm sau:", compute_quality(s.current)["overall"])
s.undo(); print("Sau undo:", len(s.current), "dòng; redo được:", s.can_redo)
s.redo()
data, name, _ = export_session(s, "xlsx")
open(name, "wb").write(data); print("Đã xuất", name, len(data), "bytes; audit:", len(s.audit), "dòng")
