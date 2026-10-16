@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  Building RDNA 3 AI Engine (rdna3_ai_engine.dll)
echo ============================================================

if not exist "bin" mkdir "bin"

where cl.exe >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Searching for Visual Studio compiler...
    if exist "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul
    ) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
    ) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
    ) else (
        echo [ERROR] MSVC cl.exe not found! Please run from 'x64 Native Tools Command Prompt for VS'.
        exit /b 1
    )
)

echo [1/2] Compiling shared library: bin\rdna3_ai_engine.dll...
cl.exe /nologo /O2 /LD /Iinclude src\rdna3_ai_engine.c /Fe:bin\rdna3_ai_engine.dll /link /DLL user32.lib advapi32.lib
if %errorlevel% neq 0 (
    echo [ERROR] Failed to compile rdna3_ai_engine.dll!
    exit /b 1
)

echo [2/3] Compiling C example: bin\c_example.exe...
cl.exe /nologo /O2 /Iinclude bindings\c\example.c bin\rdna3_ai_engine.lib /Fe:bin\c_example.exe /link user32.lib advapi32.lib
if %errorlevel% neq 0 (
    echo [ERROR] Failed to compile C example!
    exit /b 1
)

echo [3/4] Compiling Contextual Sparsity benchmark: bin\test_contextual_sparsity.exe...
cl.exe /nologo /O2 /Iinclude bindings\c\test_contextual_sparsity.c bin\rdna3_ai_engine.lib /Fe:bin\test_contextual_sparsity.exe /link user32.lib advapi32.lib
if %errorlevel% neq 0 (
    echo [ERROR] Failed to compile test_contextual_sparsity.exe!
    exit /b 1
)

echo [4/4] Compiling Token Rate benchmark: bin\benchmark_token_rate.exe...
cl.exe /nologo /O2 /Iinclude bindings\c\benchmark_token_rate.c bin\rdna3_ai_engine.lib /Fe:bin\benchmark_token_rate.exe /link user32.lib advapi32.lib
if %errorlevel% neq 0 (
    echo [ERROR] Failed to compile benchmark_token_rate.exe!
    exit /b 1
)

del /q bin\*.obj >nul 2>nul
del /q *.obj >nul 2>nul

echo ============================================================
echo [SUCCESS] Build completed successfully!
echo  - Output Library: bin\rdna3_ai_engine.dll
echo  - C Example:     bin\c_example.exe
echo  - Sparsity Test: bin\test_contextual_sparsity.exe
echo  - Token Rate:    bin\benchmark_token_rate.exe
echo ============================================================
