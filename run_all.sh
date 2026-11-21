#!/usr/bin/env bash
# Chay toan bo: backend (MongoDB gia lap) + giao dien. Admin mac dinh: admin@example.com / admin123
export USE_MOCK_DB=1 ADMIN_EMAIL=admin@example.com ADMIN_PASSWORD=admin123
python -m uvicorn backend.main:app --port 8000 &
BACK=$!; trap "kill $BACK" EXIT; sleep 4
streamlit run app.py
