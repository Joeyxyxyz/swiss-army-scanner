@echo off
REM Swiss Army Scanner web UI in MOCK mode: a simulated bench, no hardware is touched.
REM Uses its OWN library (library_mock\) and port (8766) so practice runs never land
REM in the real library's output bin, and it can run next to the real UI.
REM Use the "Sim bench" button in the page to place / clear simulated objects.
REM Serves http://127.0.0.1:8766/ only (this PC). Ctrl+C in this window stops it.
cd /d "%~dp0"
REM Python: %SCAN_PYTHON% if set (the full path to your env's python.exe),
REM else the `python` on PATH (activate the `scan` conda env first).
if not defined SCAN_PYTHON set "SCAN_PYTHON=python"
"%SCAN_PYTHON%" -m ui.web.server --mock --port 8766 --library "%~dp0library_mock" --open %*
pause
