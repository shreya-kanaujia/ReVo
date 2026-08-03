#!/usr/bin/env bash
set -euo pipefail

ROOT="${REVO_ROOT:-$HOME/ReVo}"

sudo apt-get update
sudo apt-get install -y \
  build-essential ca-certificates curl ffmpeg pkg-config \
  libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev \
  libavutil-dev libswresample-dev libswscale-dev libx265-dev \
  libgl1 libglib2.0-0

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv python install 3.12

cd "$ROOT"
if [[ -d .venv ]]; then
  mv .venv ".venv-broken-$(date +%Y%m%d-%H%M%S)"
fi

uv venv --python 3.12 --seed .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Match the working Ohio CPU environment and Ubuntu's FFmpeg 8 libraries.
python -m pip install \
  torch==2.9.0 torchvision==0.24.0 torchcodec==0.8.1 \
  --index-url https://download.pytorch.org/whl/cpu

python -m pip install \
  tqdm scipy pybind11 pillow pandas matplotlib pyyaml torchmetrics \
  aiortc aiohttp numpy opencv-python zfec einops pytorch_msssim timm

python -m pip install --no-binary av av==16.1.0

cd "$ROOT/src/receiver"
python - <<'PY'
import av
import importlib.metadata
import torch

torchcodec_version = importlib.metadata.version("torchcodec")
assert torch.__version__.startswith("2.9.0"), torch.__version__
assert torchcodec_version.startswith("0.8.1"), torchcodec_version
av.CodecContext.create("hevc", "r")
av.CodecContext.create("libx264", "w")
print("Torch/TorchCodec:", torch.__version__, torchcodec_version)
print("PyAV HEVC decoder and H.264 output encoder: OK")
PY

python receiver-3d.py --help >/dev/null
echo "AWS receiver environment is ready: $ROOT/.venv"
