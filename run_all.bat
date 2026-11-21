@echo off
REM Chay toan bo: backend (MongoDB gia lap) + giao dien. Admin mac dinh: admin@example.com / admin123
set USE_MOCK_DB=1
set ADMIN_EMAIL=admin@example.com
set ADMIN_PASSWORD=admin123
start "Backend" cmd /k python -m uvicorn backend.main:app --port 8000
timeout /t 4 >nul
streamlit run app.py
