@echo off
rem Swiss Army Scanner command line:  scan.bat [--mock] <command> [...]
rem   scan.bat --help          all commands
rem   scan.bat --mock wizard   guided run on the simulated rig (no hardware)
rem Runs from any folder: the repo is put on PYTHONPATH instead of cd-ing,
rem so relative paths you type (e.g. --library) stay relative to YOUR folder.
setlocal
set "PYTHONPATH=%~dp0;%PYTHONPATH%"
rem Python: %SCAN_PYTHON% if set (the full path to your env's python.exe),
rem else the `python` on PATH (activate the `scan` conda env first).
rem The real rig needs PySpin (FLIR Spinnaker SDK) in THAT Python: if it is
rem missing you get one line saying so - set SCAN_PYTHON to the env that has
rem it, or use --mock.
if not defined SCAN_PYTHON set "SCAN_PYTHON=python"
"%SCAN_PYTHON%" -m ui.cli %*
exit /b %ERRORLEVEL%
