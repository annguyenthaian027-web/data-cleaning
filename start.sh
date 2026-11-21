#!/usr/bin/env bash
# Chạy backend (nội bộ, cổng 8000) rồi chạy giao diện (cổng public 7860)
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
sleep 5
exec streamlit run app.py \
  --server.port "${PORT:-7860}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
