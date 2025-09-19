@echo off
setlocal
cd /d %~dp0
call .\.venv\Scripts\activate
py backtest_intraday_portfolio.py
