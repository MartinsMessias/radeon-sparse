@echo off
setlocal enabledelayedexpansion
echo ======================================================================
echo    Building RDNA 3 / RDNA 4 AI Engine + Official Benchmark Suite
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

echo [1/3] Compiling Shared Library: bin\rdna3_ai_engine.dll ...
cl /O2 /LD /W3 /I include src\rdna3_ai_engine.c /Fe:bin\rdna3_ai_engine.dll /Fo:bin\rdna3_ai_engine.obj >nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to compile rdna3_ai_engine.dll
    exit /b 1
)

echo [2/3] Compiling Official Benchmark CLI: bin\rdna3-bench.exe ...
cl /O2 /W3 /I include benchmarks\rdna3_bench.c bin\rdna3_ai_engine.lib /Fe:bin\rdna3-bench.exe /Fo:bin\rdna3_bench.obj >nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to compile rdna3-bench.exe
    exit /b 1
)

echo [3/3] Compiling C Quickstart Example: bin\use_in_c.exe ...
cl /O2 /W3 /I include examples\use_in_c.c bin\rdna3_ai_engine.lib /Fe:bin\use_in_c.exe /Fo:bin\use_in_c.obj >nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to compile use_in_c.exe
    exit /b 1
)

del /q bin\*.obj >nul 2>nul

echo [SUCCESS] Build complete:
echo   - bin\rdna3_ai_engine.dll
echo   - bin\rdna3-bench.exe
echo   - bin\use_in_c.exe
echo ======================================================================
