@echo off
REM Swiss Army Scanner web UI on the REAL rig (FLIR camera + LightCrafter over HDMI).
REM Before scanning: close the LightCrafter GUI (it holds the USB link).
REM Nothing touches the rig until you run an action in the page.
REM Do not move the camera or projector between a base and its object scans.
REM Serves http://127.0.0.1:8765/ only (this PC). Ctrl+C in this window stops it.
cd /d "%~dp0"
REM Python: %SCAN_PYTHON% if set (the full path to your env's python.exe),
REM else the `python` on PATH (activate the `scan` conda env first).
REM The real rig needs PySpin (FLIR Spinnaker SDK) in THAT Python: if it is
REM missing you get one line saying so - set SCAN_PYTHON to the env that has
REM it, or use --mock.
if not defined SCAN_PYTHON set "SCAN_PYTHON=python"
"%SCAN_PYTHON%" -m ui.web.server --open %*
pause
