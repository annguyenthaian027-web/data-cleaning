---
title: Data Cleaning Platform
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Data Cleaning Platform — bản gộp

```
app.py            Giao diện Streamlit (chạy chính)
dataclean/        Core: loader, detectors, Rule Engine, audit, undo/redo, quality score, exporter
dataclean_ai/     Lớp AI: Claude/OpenAI, che dữ liệu, kiểm duyệt rule, chế độ ngoại tuyến
backend/          FastAPI: đăng ký/đăng nhập JWT (MongoDB) + API /api/clean/*
tests_core/ tests_ai/   Kiểm thử
```

## Chạy nhanh (giao diện + backend đăng nhập, đã nối với nhau)
```
pip install -r requirements.txt
run_all.bat        (Windows)      |      ./run_all.sh        (Linux/Mac)
```
Mở http://localhost:8501. Tài khoản admin mặc định: `admin@example.com` / `admin123` (đổi trong run_all). Có thể tự đăng ký tài khoản user ngay trên giao diện.
`run_all` dùng MongoDB giả lập trong bộ nhớ, nên tài khoản mất khi tắt. Muốn lưu thật: cài MongoDB, bỏ `USE_MOCK_DB`, đặt `MONGO_URL` nếu khác `mongodb://127.0.0.1:27017`.

Chạy tách từng phần:
```
python -m uvicorn backend.main:app --port 8000      # backend, docs: http://127.0.0.1:8000/docs
streamlit run app.py                                # giao diện
```
Biến môi trường: `DATACLEAN_API_URL` (mặc định http://127.0.0.1:8000), `JWT_SECRET_KEY`, `ADMIN_EMAIL`/`ADMIN_PASSWORD`, `REQUIRE_LOGIN=0` (bỏ đăng nhập để thử nhanh), `ANTHROPIC_API_KEY` hoặc `OPENAI_API_KEY` (AI thật; không đặt thì chạy ngoại tuyến).

Luồng: đăng nhập (POST /api/login) → tải file → "Phân tích bằng AI" gọi POST /api/clean/analyze kèm token → chọn rule, xem trước/sau, áp dụng, undo/redo → tải Excel/CSV. Admin có thêm tab Quản lý user (xem, đổi quyền, xóa). /api/users chỉ admin được gọi.

## Kiểm thử
```
python demo_core.py
pytest
```
