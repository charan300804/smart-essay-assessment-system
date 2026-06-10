@echo off
echo Starting SEAS System Locally...

echo Starting Backend API...
start cmd /k "uvicorn src.api:app --reload --host 0.0.0.0 --port 8000"

echo Waiting for API to start...
timeout /t 5

echo Starting Frontend...
start cmd /k "streamlit run src/app.py"

echo System Started!
echo API is running at http://localhost:8000/docs
echo Frontend is running at http://localhost:8501
