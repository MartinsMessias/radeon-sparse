@echo off
setlocal enabledelayedexpansion
pushd "%~dp0"

echo ======================================================================
echo    RadeonSparse -- Global System Installer (Windows)
echo ======================================================================

call build_windows.bat
if %ERRORLEVEL% NEQ 0 (
    popd
    exit /b 1
)

set "INSTALL_DIR=%LOCALAPPDATA%\RadeonSparse"
if not exist "%INSTALL_DIR%\bin" mkdir "%INSTALL_DIR%\bin"
if not exist "%INSTALL_DIR%\include" mkdir "%INSTALL_DIR%\include"
if not exist "%INSTALL_DIR%\lib" mkdir "%INSTALL_DIR%\lib"

echo.
echo [1/4] Installing DLL, CLI and Headers to %INSTALL_DIR% ...
copy /y "bin\radeon_sparse.dll" "%INSTALL_DIR%\bin\" >nul
copy /y "bin\radeon-sparse-bench.exe" "%INSTALL_DIR%\bin\" >nul
copy /y "bin\radeon_sparse.lib" "%INSTALL_DIR%\lib\" >nul
copy /y "include\radeon_sparse.h" "%INSTALL_DIR%\include\" >nul
copy /y "bin\radeon_sparse.dll" "python\radeon_sparse.dll" >nul

echo [2/4] Setting global environment variable RADEON_SPARSE_LIB_PATH ...
setx RADEON_SPARSE_LIB_PATH "%INSTALL_DIR%\bin\radeon_sparse.dll" >nul
setx RADEON_SPARSE_HOME "%INSTALL_DIR%" >nul

echo [3/4] Adding %INSTALL_DIR%\bin to User PATH ...
powershell -NoProfile -Command "$p = [Environment]::GetEnvironmentVariable('Path', 'User'); if ($p -notlike '*%INSTALL_DIR%\bin*') { [Environment]::SetEnvironmentVariable('Path', $p + ';%INSTALL_DIR%\bin', 'User'); Write-Host '      Added to User PATH.' } else { Write-Host '      Already in User PATH.' }"

echo [4/4] Installing global Python package (pip install -e .) ...
python -m pip install -e . --no-deps --quiet

popd
echo.
echo ======================================================================
echo [SUCCESS] RadeonSparse is now globally installed on your system!
echo   - Global DLL Path : %INSTALL_DIR%\bin\radeon_sparse.dll
echo   - Global CLI      : radeon-sparse-bench
echo   - Global Python   : from radeon_sparse import RadeonSparseEngine
echo ======================================================================
