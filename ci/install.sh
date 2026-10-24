#!/usr/bin/env bash
# Install system and Python dependencies for the pipeline on a GitHub-hosted Ubuntu runner.
set -euo pipefail
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg libsndfile1 > /dev/null
python -m pip install --upgrade pip
python -m pip install torch==2.14.1 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r ci/requirements-ci.txt
python -m pip install --no-deps -e .
python -m songfeatures check
