#!/bin/bash
# start.sh
# Runs both FastAPI and Streamlit inside a single container instance.
# Useful for platforms that allow multiple processes, or local testing.

# 1. Start the FastAPI backend in the background
# (It will run on port 8000 inside the container)
echo "Starting FastAPI server..."
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 &

# 2. Wait a moment to ensure API is up
sleep 3

# 3. Start the Streamlit frontend in the foreground
# (It runs on port 8501. Headless flags prevent it from opening a browser window)
echo "Starting Streamlit dashboard..."
streamlit run src/dashboard.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
