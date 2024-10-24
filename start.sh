#!/bin/bash
pip install uvicorn
pip install fastapi
streamlit run main.py --server.port 8501 &
uvicorn app:app --host 0.0.0.0 --port 8000
