@echo off
REM ---------------------------------------------------------------------------
REM One-click launcher for the Eriksen Flanker task (PsychoPy 2025.2.4).
REM
REM Double-click to run. Pass --pilot for a windowed test run:
REM     run_flanker.bat --pilot
REM
REM All paths are relative to this file (%~dp0). This folder has already moved
REM twice and absolute paths broke silently each time -- do not hardcode them.
REM ---------------------------------------------------------------------------

setlocal
set "PY=%~dp0psychopy_env\Scripts\python.exe"
set "EXP=%~dp0experiment\psychopy_lastrun.py"

if not exist "%PY%" (
    echo [ERROR] PsychoPy interpreter not found:
    echo         %PY%
    echo.
    echo The psychopy_env virtualenv is missing. See CLAUDE.md in this folder.
    pause
    exit /b 1
)

if not exist "%EXP%" (
    echo [ERROR] Experiment script not found:
    echo         %EXP%
    pause
    exit /b 1
)

echo Starting Flanker task...
echo Responses: LEFT / RIGHT arrow keys. SPACE to advance. ESC to abort.
echo.

"%PY%" "%EXP%" %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo [ERROR] PsychoPy exited with code %RC%.
    pause
)

exit /b %RC%
