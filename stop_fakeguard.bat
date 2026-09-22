@echo off
echo ========================================================
echo               🛡️ Stopping FakeGuard AI
echo ========================================================
taskkill /FI "WINDOWTITLE eq FakeGuard-Backend*" /F /T >nul 2>&1
taskkill /FI "WINDOWTITLE eq FakeGuard-Frontend*" /F /T >nul 2>&1
echo All FakeGuard AI background instances stopped successfully.
pause
