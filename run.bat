@echo off
rem Double-click to start. Uses the project's own virtualenv.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" -X utf8 main.py
if errorlevel 1 pause
