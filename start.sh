#!/bin/bash

# Run Streamlit on port 8501
streamlit run main.py --server.port 8501 &

# Run FastAPI using Uvicorn on port 8000
uvicorn app:app --host 0.0.0.0 --port 8000
