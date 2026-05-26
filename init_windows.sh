#!/usr/bin/env bash

echo "Initializing environment..."

# Detect OS
OS="$(uname -s)"
echo "Detected OS: $OS"

# ---------------------------------------------------------------------------
# Determine platform and virtual environment binaries
# ---------------------------------------------------------------------------
if [[ "$OS" == MINGW* || "$OS" == CYGWIN* ]]; then
    # Windows (Git Bash)
    IS_WINDOWS=true
    PYTHON_BIN=".venv/Scripts/python.exe"
    PIP_BIN=".venv/Scripts/pip.exe"
else
    # Linux / macOS / UCloud
    IS_WINDOWS=false
    PYTHON_BIN=".venv/bin/python"
    PIP_BIN=".venv/bin/pip"
fi

# ---------------------------------------------------------------------------
# Pull latest changes
# ---------------------------------------------------------------------------
echo "Pulling latest changes from git..."
git pull

# ---------------------------------------------------------------------------
# Create virtual environment if missing
# ---------------------------------------------------------------------------
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv || python -m venv .venv
else
    echo "Virtual environment already exists."
fi

# ---------------------------------------------------------------------------
# Upgrade pip *inside* the venv
# ---------------------------------------------------------------------------
echo "Upgrading pip (inside venv)..."
"$PYTHON_BIN" -m pip install --upgrade pip

# ---------------------------------------------------------------------------
# Install project dependencies (always inside venv)
# ---------------------------------------------------------------------------
echo "Installing requirements..."
"$PIP_BIN" install -r requirements.txt

# ---------------------------------------------------------------------------
# Install Jupyter kernel only on non-Windows systems (UCloud)
# ---------------------------------------------------------------------------
if [ "$IS_WINDOWS" = false ]; then
    echo "Installing Jupyter kernel (Linux/UCloud)..."
    "$PIP_BIN" install ipykernel
    "$PYTHON_BIN" -m ipykernel install --user --name fma-venv --display-name "Python (fma-venv)"
else
    echo "Skipping Jupyter kernel install on Windows."
fi

# ---------------------------------------------------------------------------
# Set PYTHONPATH
# ---------------------------------------------------------------------------
export PYTHONPATH="$(pwd):$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"

echo "Environment setup complete."
