"""Thử đầu-cuối với khoá API thật:
    export ANTHROPIC_API_KEY=...        # hoặc OPENAI_API_KEY=...
    python -m dataclean_ai.demo_live [file.csv|file.xlsx]
Không có khoá thì chạy chế độ ngoại tuyến."""
import sys
import warnings

warnings.filterwarnings("ignore")
from dataclean import load_dataframe  # noqa: E402
from dataclean_ai import AIAdvisor, AIConfig  # noqa: E402

SAMPLE = """Mã KH,Họ tên,Email,Số điện thoại,Ngày sinh,Lương
KH01,  nguyễn văn a ,A@Gmail.com,0912345678,05/01/1990,15000000
KH02,Trần Thị B,b@@gmail,+84 912 345 679,1992-03-15,"16,000,000"
KH03,Lê Văn C,c@x.vn,912345680,31/02/1991,14000000
KH04,Phạm D,N/A,0987.654.321,20-12-1989,15500000
KH05,Hoàng E,e@x.vn,0901234567,10/10/1995,15200000
KH05,hoàng e,e@x.vn,0901234567,10/10/1995,15200000
KH06,Võ F,f@x.vn,0933333333,01/01/1993,
KH07,Đỗ G,g@x.vn,0944444444,02/02/1994,900000000
"""

if __name__ == "__main__":
    df = (load_dataframe(open(sys.argv[1], "rb").read(), sys.argv[1]) if len(sys.argv) > 1
          else load_dataframe(SAMPLE.encode("utf-8"), "mau.csv"))
    adv = AIAdvisor(AIConfig.from_env())
    print("Chế độ:", adv.mode, ("| lỗi khởi tạo: " + adv.init_error) if adv.init_error else "")
    rv = adv.review(df)
    print("degraded:", rv.degraded, "| error:", rv.error, "| usage:", rv.usage)
    print("\n== SCHEMA ==", rv.table_summary)
    for c in rv.columns:
        print(f"  {c.name:16} {c.semantic_type:12} -> {c.suggested_name:16} ({c.confidence:.2f})")
    print("\n== GIẢI THÍCH ==", rv.issues_summary)
    for f in rv.findings:
        print(f"  [{f.severity}] {f.column}: {f.explanation}")
    print("\n== RULE ==")
    for p in rv.proposals:
        r = p.rule
        print(f"  {p.status:9} {p.risk:6} {p.raw.get('action')!s:18} {p.raw.get('column')!s:16} "
              f"{'BẬT' if r and r.enabled else 'tắt'}  {' | '.join(p.notes)}")
