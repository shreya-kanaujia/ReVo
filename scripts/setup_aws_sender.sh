#!/usr/bin/env bash
set -euo pipefail

ROOT="${REVO_ROOT:-$HOME/ReVo}"
RGB_VIDEO="${RGB_VIDEO:-$ROOT/data/gt_rgb/1lSejjfNHpw_0075_S0_E728_L671_T47_R1471_B847.mp4}"

sudo apt-get update
sudo apt-get install -y \
  build-essential ca-certificates curl ffmpeg iproute2 pkg-config \
  libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev \
  libavutil-dev libswresample-dev libswscale-dev libx265-dev

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

# TorchCodec 0.8.1 supports FFmpeg 8 and is paired with Torch 2.9.
# Use CPU wheels because r6i instances do not have an NVIDIA GPU.
python -m pip install \
  torch==2.9.0 torchvision==0.24.0 torchcodec==0.8.1 \
  --index-url https://download.pytorch.org/whl/cpu

python -m pip install \
  tqdm scipy pybind11 pillow pandas matplotlib pyyaml torchmetrics \
  aiortc aiohttp numpy opencv-python zfec einops pytorch_msssim timm

# Build PyAV against Ubuntu's FFmpeg so libx265 is available to ReVo.
python -m pip install --no-binary av av==16.1.0

cd "$ROOT/src/sender"
python - <<PY
import av
import importlib.metadata
import torch
from torchcodec.decoders import VideoDecoder

torchcodec_version = importlib.metadata.version("torchcodec")
assert torch.__version__.startswith("2.9.0"), torch.__version__
assert torchcodec_version.startswith("0.8.1"), torchcodec_version
av.CodecContext.create("libx265", "w")
decoder = VideoDecoder("$RGB_VIDEO", device="cpu")
frame = decoder[0]
print("Torch/TorchCodec:", torch.__version__, torchcodec_version)
print("TorchCodec decoded frame:", tuple(frame.shape))
print("PyAV libx265 encoder: OK")
PY

python sender-3d.py --help >/dev/null
echo "AWS sender environment is ready: $ROOT/.venv"
