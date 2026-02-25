@echo off
REM Setup script for Mathematics for AI (Windows)

echo Setting up Mathematics for AI...

REM Check Python version
python --version

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install package in development mode
echo Installing package...
pip install -e ".[dev]"

REM Install pre-commit hooks
echo Installing pre-commit hooks...
pre-commit install

REM Create necessary directories
echo Creating directories...
mkdir data\generated 2>nul
mkdir data\large-datasets 2>nul
mkdir models\checkpoints 2>nul
mkdir docs\animations 2>nul

REM Create .gitkeep files
type nul > data\generated\.gitkeep
type nul > data\large-datasets\.gitkeep
type nul > models\checkpoints\.gitkeep
type nul > docs\animations\.gitkeep

echo.
echo Setup complete!
echo.
echo To get started:
echo   1. Activate the virtual environment: venv\Scripts\activate
echo   2. Run the demo: math-ai demo
echo   3. Run tests: pytest tests\
echo   4. Open a notebook: jupyter notebook notebooks\basics\01_introduction.ipynb
echo.
