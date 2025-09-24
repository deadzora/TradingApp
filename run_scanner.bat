@echo off
setlocal enabledelayedexpansion

REM ---- load .env into current session (lines like KEY=VALUE)
if not exist .env (
  echo [.env] file not found. Create .env with APCA_API_KEY_ID and APCA_API_SECRET_KEY.
  pause
  exit /b 1
)

for /f "usebackq tokens=1* delims==" %%A in (".env") do (
  set "line=%%A"
  REM skip empty lines and comments
  if not "!line!"=="" (
    if not "!line:~0,1!"=="#" (
      set "key=%%A"
      set "val=%%B"
      REM trim possible surrounding spaces
      set "key=!key: =!"
      set "val=!val: =!"
      setx_temp=1
      set "!key!=!val!"
    )
  )
)

REM (optional) show masked key for verification
if defined APCA_API_KEY_ID (
  set "k=!APCA_API_KEY_ID!"
  set "k4=!k:~0,4!"
  echo [boot] APCA_API_KEY_ID masked: !k4!******
) else (
  echo [boot] APCA_API_KEY_ID not set in .env
)

REM activate venv (adjust path if different)
call .\.venv\Scripts\activate

REM run scanner in venv (this process inherits the env vars we just set)
python scanner.py

endlocal
