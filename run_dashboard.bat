@echo off
setlocal
cd /d %~dp0
cd dashboard
call ..\.venv\Scripts\activate
set RB_DASH_LIMIT=300
start "" http://localhost:8080
py app.py
