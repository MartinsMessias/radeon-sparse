@echo off
setlocal enabledelayedexpansion
echo ======================================================================
echo    Building RadeonSparse Engine + Official Benchmark Suite
echo ======================================================================

where cl >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if exist "!VSWHERE!" (
        for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
            set "VS_PATH=%%i"
        )
    )
    if defined VS_PATH (
        if exist "!VS_PATH!\VC\Auxiliary\Build\vcvars64.bat" (
            call "!VS_PATH!\VC\Auxiliary\Build\vcvars64.bat" >nul
        )
    )
)

where cl >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] MSVC compiler cl.exe not found in PATH or via vswhere.
    exit /b 1
)

if not exist bin mkdir bin

echo [1/3] Compiling Shared Library: bin\radeon_sparse.dll ...
cl /O2 /LD /W3 /I include src\radeon_sparse.c /Fe:bin\radeon_sparse.dll /Fo:bin\radeon_sparse.obj >nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to compile radeon_sparse.dll
    exit /b 1
)

echo [2/3] Compiling Official Benchmark CLI: bin\radeon-sparse-bench.exe ...
cl /O2 /W3 /I include benchmarks\radeon_sparse_bench.c bin\radeon_sparse.lib /Fe:bin\radeon-sparse-bench.exe /Fo:bin\radeon_sparse_bench.obj >nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to compile radeon-sparse-bench.exe
    exit /b 1
)

echo [3/3] Compiling C Quickstart Example: bin\use_in_c.exe ...
cl /O2 /W3 /I include examples\use_in_c.c bin\radeon_sparse.lib /Fe:bin\use_in_c.exe /Fo:bin\use_in_c.obj >nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to compile use_in_c.exe
    exit /b 1
)

del /q bin\*.obj >nul 2>nul

echo [SUCCESS] Build complete:
echo   - bin\radeon_sparse.dll
echo   - bin\radeon-sparse-bench.exe
echo   - bin\use_in_c.exe
echo ======================================================================
