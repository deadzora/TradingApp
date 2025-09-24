@echo off
setlocal enabledelayedexpansion

if not exist .env (
  echo [.env] file not found. Create .env with APCA_API_KEY_ID and APCA_API_SECRET_KEY.
  pause
  exit /b 1
)

for /f "usebackq tokens=1* delims==" %%A in (".env") do (
  set "k=%%A"
  set "v=%%B"
  if not "!k!"=="" if not "!k:~0,1!"=="#" (
    set "!k!=!v!"
  )
)

REM force canonical market data host (no /v2)
set APCA_API_DATA_URL=https://data.alpaca.markets

REM sanity print masked
set "k=%APCA_API_KEY_ID%"
echo [boot] APCA_API_KEY_ID masked: %k:~0,4%******  data=%APCA_API_DATA_URL%

call .\.venv\Scripts\activate
python run_loop_multi.py

endlocal