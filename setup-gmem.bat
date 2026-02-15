@echo off
REM setup-gmem.bat — Install gmem as a global command on Windows
REM Run this once after setting up the venv

SET SCRIPT_DIR=%~dp0
SET VENV_PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe
SET GMEM_SCRIPT=%SCRIPT_DIR%gmem.py

REM Check venv exists
IF NOT EXIST "%VENV_PYTHON%" (
    echo ❌ Virtual environment not found. Run these first:
    echo    python -m venv .venv
    echo    .venv\Scripts\activate
    echo    pip install -r requirements.txt
    exit /b 1
)

REM Create gmem.bat in venv Scripts (which should be on PATH when venv is active)
SET GMEM_BAT=%SCRIPT_DIR%.venv\Scripts\gmem.bat
echo @echo off > "%GMEM_BAT%"
echo "%VENV_PYTHON%" "%GMEM_SCRIPT%" %%* >> "%GMEM_BAT%"

echo ✅ gmem command installed!
echo    Make sure your venv is activated, then use: gmem --help
echo.
echo    Or add this to your PATH for global access:
echo    %SCRIPT_DIR%.venv\Scripts
