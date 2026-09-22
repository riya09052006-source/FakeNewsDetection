@echo off
echo ========================================================
echo               🛡️ FakeGuard AI Launcher
echo ========================================================
cd /d "%~dp0"

echo [1/2] Starting Backend API Server (FastAPI on Port 8000)...
start "FakeGuard-Backend" cmd /k ".\venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000"

echo [2/2] Starting Frontend UI (React + Vite on Port 5173)...
cd frontend
start "FakeGuard-Frontend" cmd /k "npm run dev"

echo.
echo ========================================================
echo Services started!
echo Backend API:       http://127.0.0.1:8000
echo Interactive Docs:  http://127.0.0.1:8000/docs
echo Web Application:   http://localhost:5173
echo ========================================================
pause
