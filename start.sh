#!/bin/bash
pip install uvicorn
pip install fastapi
pip install gunicorn

# Start Streamlit on a different port (8501)
streamlit run main.py --server.port 8501 &

# Start FastAPI using Uvicorn on port 8000
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:8000
