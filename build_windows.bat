@echo off
setlocal
echo ========================================================
echo  Building rdna3-ai-engine for Windows (MSVC x64)
echo ========================================================

REM Find MSVC vcvars64.bat if not already in dev prompt
where cl.exe >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    if exist "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    ) else (
        echo [ERROR] MSVC cl.exe not found! Please run inside Visual Studio Developer Prompt.
        exit /b 1
    )
)

echo [*] Compiling dynamic link library (rdna3_ai_engine.dll)...
cl /LD /MD /O2 /I include src\rdna3_ai_engine.c /Fe:rdna3_ai_engine.dll /link /OUT:rdna3_ai_engine.dll
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] DLL compilation failed!
    exit /b %ERRORLEVEL%
)

echo [*] Compiling C test executable (use_in_c.exe)...
cl /MD /O2 /I include examples\use_in_c.c rdna3_ai_engine.lib /Fe:use_in_c.exe
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] C example compilation failed!
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================================
echo  BUILD SUCCESSFUL!
echo  Generated: rdna3_ai_engine.dll and use_in_c.exe
echo ========================================================
