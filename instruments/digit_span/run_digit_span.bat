@echo off
REM ---------------------------------------------------------------------------
REM One-click launcher for the Digit Span administration tool.
REM
REM Double-click to open the admin page in the default browser. No server and
REM no Python needed -- audio is loaded over file:// by relative path.
REM
REM All paths are relative to this file (%~dp0). The HTML and digit_span_audio/
REM MUST stay siblings or every stimulus 404s -- see CLAUDE.md in this folder.
REM ---------------------------------------------------------------------------

setlocal
set "PAGE=%~dp0digit_span_admin.html"
set "AUDIO=%~dp0digit_span_audio"

if not exist "%PAGE%" (
    echo [ERROR] Admin page not found:
    echo         %PAGE%
    pause
    exit /b 1
)

if not exist "%AUDIO%" (
    echo [WARNING] Audio folder not found:
    echo           %AUDIO%
    echo.
    echo The tool will open but every digit sequence will fail to load.
    echo Regenerate with: python generate_digit_span.py
    echo.
    pause
)

start "" "%PAGE%"
exit /b 0
