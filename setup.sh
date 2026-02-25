# Setup script for Mathematics for AI

#!/bin/bash

echo "Setting up Mathematics for AI..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install package in development mode
echo "Installing package..."
pip install -e ".[dev]"

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "Creating directories..."
mkdir -p data/generated
mkdir -p data/large-datasets
mkdir -p models/checkpoints
mkdir -p docs/animations

# Create .gitkeep files
touch data/generated/.gitkeep
touch data/large-datasets/.gitkeep
touch models/checkpoints/.gitkeep
touch docs/animations/.gitkeep

echo ""
echo "Setup complete!"
echo ""
echo "To get started:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run the demo: math-ai demo"
echo "  3. Run tests: pytest tests/"
echo "  4. Open a notebook: jupyter notebook notebooks/basics/01_introduction.ipynb"
echo ""
