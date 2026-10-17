@echo off
setlocal enabledelayedexpansion
echo ======================================================================
echo    RadeonSparse — Global System Installer (Windows)
echo ======================================================================

:: 1. Build binaries if not already built
call "%~dp0build_windows.bat"
if %ERRORLEVEL% NEQ 0 exit /b 1

:: 2. Create global install directory in %LOCALAPPDATA%\RadeonSparse
set "INSTALL_DIR=%LOCALAPPDATA%\RadeonSparse"
if not exist "%INSTALL_DIR%\bin" mkdir "%INSTALL_DIR%\bin"
if not exist "%INSTALL_DIR%\include" mkdir "%INSTALL_DIR%\include"
if not exist "%INSTALL_DIR%\lib" mkdir "%INSTALL_DIR%\lib"

echo.
echo [1/4] Installing DLL, CLI & Headers to %INSTALL_DIR% ...
copy /y "%~dp0bin\radeon_sparse.dll" "%INSTALL_DIR%\bin\" >nul
copy /y "%~dp0bin\radeon-sparse-bench.exe" "%INSTALL_DIR%\bin\" >nul
copy /y "%~dp0bin\radeon_sparse.lib" "%INSTALL_DIR%\lib\" >nul
copy /y "%~dp0include\radeon_sparse.h" "%INSTALL_DIR%\include\" >nul

:: Also copy DLL into python/ folder so pip package bundles it directly
copy /y "%~dp0bin\radeon_sparse.dll" "%~dp0python\radeon_sparse.dll" >nul

echo [2/4] Setting global environment variable RADEON_SPARSE_LIB_PATH ...
setx RADEON_SPARSE_LIB_PATH "%INSTALL_DIR%\bin\radeon_sparse.dll" >nul
setx RADEON_SPARSE_HOME "%INSTALL_DIR%" >nul

echo [3/4] Adding %INSTALL_DIR%\bin to User PATH ...
powershell -NoProfile -Command "$p = [Environment]::GetEnvironmentVariable('Path', 'User'); if ($p -notlike '*%INSTALL_DIR%\bin*') { [Environment]::SetEnvironmentVariable('Path', $p + ';%INSTALL_DIR%\bin', 'User'); Write-Host '      Added to User PATH.' } else { Write-Host '      Already in User PATH.' }"

echo [4/4] Installing global Python package (pip install -e .) ...
python -m pip install -e "%~dp0" --quiet

echo.
echo ======================================================================
echo [SUCCESS] RadeonSparse is now globally installed on your system!
echo   - Global DLL Path : %INSTALL_DIR%\bin\radeon_sparse.dll
echo   - Global CLI      : radeon-sparse-bench
echo   - Global Python   : from radeon_sparse import RadeonSparseEngine
echo ======================================================================
