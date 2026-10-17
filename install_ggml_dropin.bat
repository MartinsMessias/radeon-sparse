@echo off
setlocal enabledelayedexpansion
pushd "%~dp0"

echo ======================================================================
echo    RadeonSparse -- GGML / Vulkan Drop-In Proxy Installer
echo ======================================================================

if "%~1"=="" (
    echo Usage:
    echo   install_ggml_dropin.bat "C:\Path\To\Ollama_or_LlamaCpp_Folder"
    echo.
    echo Common target folders:
    echo   - Ollama    : "%LOCALAPPDATA%\Programs\Ollama\lib\ollama"
    echo   - LM Studio : "%LOCALAPPDATA%\Programs\LM Studio\resources"
    echo   - llama.cpp : Your llama-server.exe / ggml-vulkan.dll folder
    popd
    exit /b 1
)

set "TARGET_DIR=%~1"
if not exist "%TARGET_DIR%\ggml-vulkan.dll" (
    echo [ERROR] ggml-vulkan.dll not found in "%TARGET_DIR%"
    popd
    exit /b 1
)

if not exist "bin\radeon_sparse.dll" (
    call build_windows.bat
)

if not exist "%TARGET_DIR%\ggml-vulkan.dll.original.bak" (
    echo [1/2] Backing up original ggml-vulkan.dll to ggml-vulkan.dll.original.bak ...
    copy /y "%TARGET_DIR%\ggml-vulkan.dll" "%TARGET_DIR%\ggml-vulkan.dll.original.bak" >nul
) else (
    echo [1/2] Original backup ggml-vulkan.dll.original.bak already exists.
)

echo [2/2] Installing RadeonSparse Drop-In Proxy as "%TARGET_DIR%\ggml-vulkan.dll" ...
copy /y "bin\radeon_sparse.dll" "%TARGET_DIR%\ggml-vulkan.dll" >nul

popd
echo ======================================================================
echo [SUCCESS] Drop-In Proxy installed in "%TARGET_DIR%"!
echo To restore original at any time, copy ggml-vulkan.dll.original.bak back to ggml-vulkan.dll.
echo ======================================================================
