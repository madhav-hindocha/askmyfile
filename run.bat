@echo off
title AskMyFile
cd /d "%~dp0"

echo ============================================
echo   AskMyFile - starting up, please wait...
echo ============================================
echo.

REM --- Check if using Cloud AI (OpenRouter or Groq) ---
set USE_CLOUD=0
if exist ".env" (
    for /f "usebackq eol=# tokens=1,2 delims==" %%a in (".env") do (
        if /i "%%a"=="OPENROUTER_API_KEY" if not "%%b"=="" set USE_CLOUD=1
        if /i "%%a"=="GROQ_API_KEY" if not "%%b"=="" set USE_CLOUD=1
    )
)

if "%USE_CLOUD%"=="1" echo Using Cloud AI (OpenRouter/Groq). Skipping Ollama check.
if "%USE_CLOUD%"=="1" goto :skip_ollama_install
echo.

REM --- Check Python is installed (auto-installs via winget if missing) ---
where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found -- installing it automatically, please wait...
    winget install -e --id Python.Python.3.13 --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo.
        echo Automatic install failed. Please install Python manually from
        echo https://www.python.org/downloads/ and tick "Add python.exe to PATH".
        echo Then double-click run.bat again.
        echo.
        pause
        exit /b 1
    )
    echo.
    echo ============================================
    echo   Python was just installed successfully!
    echo   Please close this window and double-click
    echo   run.bat ONCE MORE to continue setup.
    echo ============================================
    echo.
    pause
    exit /b 0
)

REM --- Check Ollama is installed (auto-installs via winget if missing) ---
where ollama >nul 2>nul
if errorlevel 1 (
    echo Ollama was not found -- installing it automatically, please wait...
    winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo.
        echo Automatic install failed. Please install Ollama manually from
        echo https://ollama.com then double-click run.bat again.
        echo.
        pause
        exit /b 1
    )
    echo.
    echo ============================================
    echo   Ollama was just installed successfully!
    echo   Please close this window and double-click
    echo   run.bat ONCE MORE to continue setup.
    echo ============================================
    echo.
    pause
    exit /b 0
)

:skip_ollama_install
REM --- Check Tesseract OCR (used for scanned PDFs and images) ---
if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" goto :tesseract_ok
where tesseract >nul 2>nul
if errorlevel 1 (
    echo Tesseract OCR not found -- trying to install it automatically...
    winget install -e --id UB-Mannheim.TesseractOCR --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo.
        echo Could not auto-install Tesseract OCR. Scanned PDFs and images
        echo won't be readable until you install it manually from:
        echo   https://github.com/UB-Mannheim/tesseract/wiki
        echo Everything else works normally -- continuing...
        echo.
    )
)
:tesseract_ok

REM --- Create the virtual environment only the first time ---
set FIRST_RUN=0
if not exist "venv\" (
    set FIRST_RUN=1
    echo First-time setup: creating a virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

REM --- Install/upgrade dependencies (fast no-op if already installed) ---
echo Checking dependencies...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo.
    echo Dependency installation failed. See the error above for details.
    echo.
    pause
    exit /b 1
)

if "%USE_CLOUD%"=="1" goto :skip_ollama_pull
REM --- Make sure the AI model is available locally ---
REM Reads OLLAMA_MODEL from .env if present, otherwise uses the default.
set OLLAMA_MODEL=llama3.2:1b
if exist ".env" (
    for /f "usebackq eol=# tokens=1,2 delims==" %%a in (".env") do (
        if /i "%%a"=="OLLAMA_MODEL" set OLLAMA_MODEL=%%b
    )
)
echo Checking AI model %OLLAMA_MODEL% (downloads once, may take a while the first time)...
ollama pull %OLLAMA_MODEL%
:skip_ollama_pull

REM --- Offer to create a Desktop shortcut, first run only ---
if "%FIRST_RUN%"=="1" (
    echo.
    echo Tip: the Desktop icon below launches AskMyFile with NO terminal
    echo window -- it runs quietly in the background and only your
    echo browser page opens. This window you're looking at now is just
    echo for this first manual run.
    set /p CREATE_SHORTCUT="Create a desktop icon for AskMyFile? (y/n): "
    if /i "%CREATE_SHORTCUT%"=="y" call create_desktop_shortcut.bat
)

if "%USE_CLOUD%"=="1" goto :skip_ollama_serve
REM --- Start Ollama in the background if it isn't already running ---
start "" /min ollama serve
:skip_ollama_serve

REM --- Open the browser after a short delay so the server has time to start ---
start "" cmd /c "timeout /t 4 >nul && start http://127.0.0.1:5000"

echo.
echo ============================================
echo   Server is starting! Your browser will open
echo   automatically in a few seconds.
echo   Keep this window open while using the app.
echo ============================================
echo.

python app.py

pause
