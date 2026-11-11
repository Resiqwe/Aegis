@echo off
cd /d "%~dp0"
if exist ".\.venv\Scripts\pythonw.exe" (
    start "" ".\.venv\Scripts\pythonw.exe" -X utf8 launcher.py
) else (
    start "" pythonw -X utf8 launcher.py
)
exit /b 0
