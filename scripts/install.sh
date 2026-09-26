#!/bin/bash
# Install dependencies and package

# Install system dependencies (if needed)
# apt-get update && apt-get install -y ffmpeg

# Install Python package
pip install -e ".[dev]"

# Download models (will be done on first run)
# python -c "import whisper; whisper.load_model('tiny')"