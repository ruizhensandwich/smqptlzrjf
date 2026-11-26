#!/usr/bin/env bash
# One-time setup on a Linux GPU machine (GCP VM, RunPod pod, ...), run in the repo root:
#
#   bash setup.sh                 # .venv with torch (CUDA wheel), dev packages, pesq 0.0.4
#   source .venv/bin/activate     # every new shell
#
# Data is not in the repository (students' recordings). Copy it from your PC:
#   scp -r data/train data/validation <host>:~/smqptlzrjf/data/
# or point DATA_ROOT at a folder that holds train/ and validation/.
set -euo pipefail
cd "$(dirname "$0")"

SUDO=""
if [ "$(id -u)" != "0" ] && command -v sudo >/dev/null 2>&1; then SUDO="sudo"; fi
if command -v apt-get >/dev/null 2>&1; then
  missing=""
  for pkg in build-essential python3-venv python3-dev tmux; do
    dpkg -s "$pkg" >/dev/null 2>&1 || missing="$missing $pkg"
  done
  if [ -n "$missing" ]; then
    $SUDO apt-get update -y && $SUDO apt-get install -y $missing
  fi
fi

PY="${PYTHON:-python3}"
[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install -q -U pip wheel setuptools
python -m pip install -q torch                 # Linux wheels on PyPI include CUDA
python -m pip install -q "numpy>=1.26,<3" "scipy>=1.12,<2" "soundfile>=0.12,<1" "einops>=0.7,<1" "rotary-embedding-torch==0.6.1" \
  "pandas>=2.1,<4" "pytest>=8" "matplotlib>=3.8" Cython
python -m pip install -q --no-build-isolation pesq==0.0.4   # C extension built against this numpy

git config core.hooksPath .githooks            # secret scan before every push

python - <<'EOF'
import torch, pesq, soundfile, numpy
print(f"torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}"
      + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
print(f"numpy {numpy.__version__}, soundfile {soundfile.__version__}, pesq ok")
EOF
if [ -f data/train/samples.csv ] && [ -f data/validation/samples.csv ]; then
  echo "data: found in data/"
else
  echo "data: NOT found. Copy train/ and validation/ into data/ (see the top of this file)."
fi
