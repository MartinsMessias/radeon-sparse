@echo off
setlocal enabledelayedexpansion
echo ========================================================
echo  Building rdna3-ai-engine for Windows (MSVC x64)
echo ========================================================

REM Check if cl.exe is already initialized in path
where cl.exe >nul 2>nul
if %ERRORLEVEL% EQU 0 goto :compile

REM Locate vcvars64.bat via official vswhere.exe tool
set "VS_PATH="
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
    for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        set "VS_PATH=%%i"
    )
)

if defined VS_PATH (
    if exist "!VS_PATH!\VC\Auxiliary\Build\vcvars64.bat" (
        echo [*] Initializing MSVC environment from: !VS_PATH!
        call "!VS_PATH!\VC\Auxiliary\Build\vcvars64.bat"
        goto :compile
    )
)

REM Fallback common locations
if exist "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
    goto :compile
)
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    goto :compile
)

echo [ERROR] MSVC cl.exe not found! Please run inside Visual Studio Developer Prompt.
exit /b 1

:compile
echo [*] Compiling dynamic link library (rdna3_ai_engine.dll)...
cl /LD /MD /O2 /W4 /wd4100 /wd4204 /wd4152 /I include src\rdna3_ai_engine.c /Fe:rdna3_ai_engine.dll /link /OUT:rdna3_ai_engine.dll
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] DLL compilation failed!
    exit /b %ERRORLEVEL%
)

echo [*] Compiling C test executable (use_in_c.exe)...
cl /MD /O2 /W4 /wd4100 /I include examples\use_in_c.c rdna3_ai_engine.lib /Fe:use_in_c.exe
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] C example compilation failed!
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================================
echo  BUILD SUCCESSFUL!
echo  Generated: rdna3_ai_engine.dll and use_in_c.exe
echo ========================================================
