@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" -Interactive %*
set "installExit=%ERRORLEVEL%"
echo.
if not "%installExit%"=="0" echo Installation failed. Read the error above before retrying.
pause
exit /b %installExit%
