"""Giao diện Streamlit — chạy: streamlit run app.py
Dùng trực tiếp core (dataclean) + lớp AI (dataclean_ai). Không có khoá API thì AI tự chạy chế độ ngoại tuyến."""
import warnings
warnings.filterwarnings("ignore")
import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from dataclean import CleaningSession, FileLoadError, Rule, analyze, export_session, load_dataframe
from dataclean_ai import AIAdvisor, AIConfig

API_URL = os.getenv("DATACLEAN_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUIRE_LOGIN = os.getenv("REQUIRE_LOGIN", "1") == "1"   # REQUIRE_LOGIN=0: bỏ qua đăng nhập (chỉ để chạy thử)


def api(method, path, token=None, **kw):
    """Gọi backend. Trả (ok, dữ liệu | thông báo lỗi)."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = requests.request(method, API_URL + path, headers=headers, timeout=120, **kw)
    except requests.RequestException:
        return False, f"Không kết nối được backend tại {API_URL}. Hãy chạy backend trước (xem README)."
    try:
        data = r.json()
    except ValueError:
        data = {}
    if r.ok:
        return True, data
    return False, data.get("detail") or f"Lỗi {r.status_code}"

st.set_page_config(page_title="Data Cleaning AI Platform", layout="wide")
st.title("🧹 Nền tảng làm sạch dữ liệu Excel/CSV có AI gợi ý")


@st.cache_resource
def get_advisor():
    return AIAdvisor(AIConfig.from_env())


adv = get_advisor()

# ---------------- Đăng nhập / đăng ký (qua backend FastAPI) ----------------
user = st.session_state.get("user")
if REQUIRE_LOGIN and not user:
    st.subheader("Đăng nhập để sử dụng")
    tl, tr = st.tabs(["Đăng nhập", "Đăng ký"])
    with tl:
        em = st.text_input("Email", key="li_em")
        pw = st.text_input("Mật khẩu", type="password", key="li_pw")
        if st.button("Đăng nhập", type="primary"):
            ok, d = api("POST", "/api/login", json={"email": em, "password": pw})
            if ok:
                st.session_state.user = d
                st.rerun()
            st.error(d)
    with tr:
        fn = st.text_input("Họ tên", key="rg_fn")
        em2 = st.text_input("Email", key="rg_em")
        pw2 = st.text_input("Mật khẩu (tối thiểu 6 ký tự)", type="password", key="rg_pw")
        if st.button("Tạo tài khoản"):
            ok, d = api("POST", "/api/register", json={"full_name": fn, "email": em2, "password": pw2})
            st.success("Đăng ký thành công, hãy đăng nhập.") if ok else st.error(d)
    st.stop()

token = user["access_token"] if user else None
is_admin = bool(user and user.get("role") == "admin")
if user:
    st.sidebar.success(f"👤 {user['full_name']} ({user['role']})")
    if st.sidebar.button("Đăng xuất"):
        st.session_state.clear()
        st.rerun()
st.sidebar.header("Dữ liệu")
up = st.sidebar.file_uploader("Chọn file Excel hoặc CSV", type=["csv", "xlsx", "xls"])
st.sidebar.caption(f"Chế độ AI: **{adv.mode}**" + (" (đặt ANTHROPIC_API_KEY hoặc OPENAI_API_KEY để dùng LLM thật)" if adv.mode == "offline" else ""))
if not up:
    st.info("👈 Tải một file CSV/Excel ở thanh bên để bắt đầu.")
    st.stop()

key = (up.name, up.size)
if st.session_state.get("key") != key:
    try:
        df0 = load_dataframe(up.getvalue(), up.name)
    except FileLoadError as e:
        st.error(str(e)); st.stop()
    st.session_state.update(key=key, sess=CleaningSession(df0, up.name), ai=None)
sess: CleaningSession = st.session_state.sess
df = sess.current
rep = analyze(df)
q = rep["quality"]

c = st.columns(5)
c[0].metric("Số dòng", len(df)); c[1].metric("Số cột", df.shape[1])
c[2].metric("Ô thiếu", int(df.isnull().sum().sum()))
c[3].metric("Lỗi phát hiện", sum(rep["issue_summary"].values()))
c[4].metric("Điểm chất lượng", f'{q["overall"]} ({q["grade"]})')

names = ["📄 Dữ liệu", "🤖 AI gợi ý & làm sạch", "📊 Báo cáo chất lượng", "🕘 Phiên bản & audit", "📥 Xuất file"]
if is_admin:
    names.append("👥 Quản lý user")
tabs = st.tabs(names)
t1, t2, t3, t4, t5 = tabs[:5]

with t1:
    st.dataframe(df.head(200), use_container_width=True)
    st.caption("Kiểu cột suy luận: " + ", ".join(f"{k}: {v}" for k, v in rep["types"].items()))

with t2:
    if st.button("✨ Phân tích bằng AI", type="primary"):
        with st.spinner("Đang phân tích..."):
            res = None
            if token:   # đã đăng nhập: phân tích qua API backend (có xác thực)
                ok, d = api("POST", "/api/clean/analyze", token,
                            files={"file": ("current.csv", df.to_csv(index=False).encode("utf-8-sig"), "text/csv")})
                if ok:
                    res = d["ai"]
                else:
                    st.warning(f"Gọi backend lỗi ({d}), chuyển sang phân tích cục bộ.")
            st.session_state.ai = res or adv.review(df, rep).to_dict()
    rv = st.session_state.get("ai")
    if rv is None:
        st.info("Bấm “Phân tích bằng AI” để nhận giải thích lỗi và rule đề xuất.")
    else:
        if rv["degraded"]:
            st.warning(f"Đang dùng gợi ý rule-based (ngoại tuyến). {rv.get('error') or ''}")
        st.write(rv["table_summary"]); st.write(rv["issues_summary"])
        for f in rv["findings"]:
            st.markdown(f"- **[{f['severity']}] {f['column'] or 'Toàn bảng'}**: {f['explanation']}")
        st.subheader("Rule đề xuất — tick để chọn, AI chỉ gợi ý, bạn quyết định")
        chosen = []
        for i, p in enumerate((rv.get("translation") or {}).get("proposals", [])):
            if not p.get("rule"):
                continue
            r = Rule.from_dict(p["rule"])
            ok = st.checkbox(f"{r.name or r.action} · cột: {r.column or 'tất cả'} · rủi ro: {p['risk']}",
                             value=r.enabled, key=f"r{i}", help=r.reason or " | ".join(p.get("notes", [])))
            if ok:
                chosen.append(r)
        if chosen:
            pv = sess.preview(chosen)
            a, b, d = st.columns(3)
            a.metric("Điểm trước", pv["quality_before"]); b.metric("Điểm sau (xem trước)", pv["quality_after"])
            d.metric("Số thay đổi", pv["stats"].get("changes", 0))
            st.write("Mẫu thay đổi (trước → sau)")
            st.dataframe(pd.DataFrame(pv["changes_sample"]), use_container_width=True)
            if st.button("✅ Áp dụng các rule đã chọn"):
                sess.apply(chosen, "Làm sạch bằng gợi ý AI"); st.session_state.ai = None; st.rerun()

with t3:
    st.json(q["dimensions"])
    miss = pd.DataFrame(q["per_column"])
    st.plotly_chart(px.bar(miss, x="column", y="missing", title="Số ô thiếu theo cột"), use_container_width=True)
    st.plotly_chart(px.bar(miss, x="column", y="issues", title="Số lỗi theo cột"), use_container_width=True)

with t4:
    u, r_ = st.columns(2)
    if u.button("↩️ Hoàn tác", disabled=not sess.can_undo): sess.undo(); st.rerun()
    if r_.button("↪️ Làm lại", disabled=not sess.can_redo): sess.redo(); st.rerun()
    st.dataframe(pd.DataFrame(sess.history()), use_container_width=True)
    st.subheader("Audit log"); st.dataframe(sess.audit.to_dataframe(), use_container_width=True)

with t5:
    for fmt, label in (("xlsx", "Excel (dữ liệu sạch + báo cáo + audit log)"), ("csv", "CSV (UTF-8)")):
        data, name, mime = export_session(sess, fmt)
        st.download_button(f"📥 Tải {label}", data, file_name=name, mime=mime)

if is_admin:
    with tabs[5]:
        ok, d = api("GET", "/api/users", token)
        if not ok:
            st.error(d)
        else:
            st.dataframe(pd.DataFrame(d["users"]), use_container_width=True)
            opts = {f"{u['full_name']} <{u['email']}>": u for u in d["users"]}
            pick = st.selectbox("Chọn người dùng", list(opts))
            u = opts[pick]
            a, b = st.columns(2)
            new_role = a.selectbox("Quyền", ["user", "admin"], index=0 if u["role"] == "user" else 1)
            if a.button("Cập nhật quyền"):
                ok, m = api("PUT", f"/api/users/{u['id']}/role", token, json={"role": new_role})
                st.success("Đã cập nhật") if ok else st.error(m)
                st.rerun() if ok else None
            if b.button("🗑️ Xóa người dùng"):
                ok, m = api("DELETE", f"/api/users/{u['id']}", token)
                st.success("Đã xóa") if ok else st.error(m)
                st.rerun() if ok else None
