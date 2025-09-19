@echo off
setlocal
cd /d %~dp0
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

if not exist .venv (
  py -m venv .venv
)
call .\.venv\Scripts\activate

py -m pip install --upgrade pip
py -m pip install -r requirements.txt

if "%APCA_API_KEY_ID%"=="" (
  echo Set your Alpaca keys with:
  echo   set APCA_API_KEY_ID=your_key
  echo   set APCA_API_SECRET_KEY=your_secret
  echo and re-run this .bat
  pause
  exit /b 1
)

REM === Logging section added ===
set "LOGFILE=logs\riskybot_run.log"
if not exist logs mkdir logs

echo Starting RiskyBot, logging to %LOGFILE% ...
py -u run_loop_multi.py > "%LOGFILE%" 2>&1

echo.
echo Finished. Check %LOGFILE% for details.
pause
