<div align="center">

# ReVo: A Cross-Layer Reliable Volumetric Videoconferencing System

[![Project Website](https://img.shields.io/badge/Project-Website-blue.svg)](https://umassos.github.io/revo-website/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.27441-b31b1b.svg)](https://arxiv.org/abs/2604.27441)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow.svg)](https://huggingface.co/umass-lass/ReVo)

**Ankur Aditya**<sup>1*</sup>, **Diptyaroop Maji**<sup>1*</sup>, **Lingdong Wang**<sup>1</sup>, **Bhavya Ramakrishna**<sup>2</sup>, **Ramesh Sitaraman**<sup>1,3</sup>, **Prashant Shenoy**<sup>1</sup>

<sup>1</sup>University of Massachusetts Amherst &nbsp;&nbsp;&nbsp; <sup>2</sup>Dolby Labs &nbsp;&nbsp;&nbsp; <sup>3</sup>Akamai Tech
<br>
<sup>*</sup> *Student authors with equal contribution*

</div>

ReVo streams synchronized **RGB + depth** video between two machines in
real time using WebRTC DataChannels.  It supports traditional codecs (H.264,
H.265) and the neural DCVC-RT codec, with forward error correction on I-frames
and deadline-based frame assembly on the receiver.

---

## Repository layout

```
FAQ.md
scripts/
├── generate_frame_masks.py     Builds per-video frame-corruption masks from receiver logs
└── download_checkpoints.sh     Downloads pre-trained LossRec checkpoints from Hugging Face
src/
├── signalling_server.py        WebRTC signaling server (run on any machine reachable by sender and receiver)
├── sender/
│   ├── sender-3d.py            Sender — reads video files, encodes, streams over WebRTC
│   ├── run_sender_eval.py      Batch evaluation script (iterates over traces × videos)
│   ├── run_loss_trace.py       Network emulator — applies tc/netem from a trace file
│   ├── H264_wrapper.py         H.264 codec interface
│   ├── H265_wrapper.py         H.265 / HEVC codec interface
│   ├── DCVCRT_wrapper.py       DCVC-RT neural codec interface
│   ├── trace_map.txt           Maps (category, video) → trace file for batch eval
│   ├── traces/                 Network trace files organized by category (wifi/, cell/, …)
│   └── sender.md               Detailed sender documentation
├── receiver/
│   ├── receiver-3d.py          Receiver — reassembles frames, decodes, saves MP4
│   ├── run_receiver_eval.py    Batch evaluation service (pairs with run_sender_eval.py)
│   ├── H264_wrapper.py         H.264 codec interface
│   ├── H265_wrapper.py         H.265 / HEVC codec interface
│   ├── DCVCRT_wrapper.py       DCVC-RT neural codec interface
│   └── receiver.md             Detailed receiver documentation
└── lossrec/
    ├── rgb/                    RGB loss recovery model and inference script
    ├── depth/                  Depth loss recovery model and inference script
    └── lossrec.md              Usage, flag reference, and checkpoint documentation
```

> **Neural loss recovery training details coming soon.**

---

## How it works

```
  Sender machine                Signaling server            Receiver machine
  ─────────────                 ────────────────            ────────────────
  sender-3d.py  ──── SDP ────►  signalling_server.py  ◄──── receiver-3d.py
                ◄─── SDP ────                          ────►
                                                              │
  [WebRTC DataChannels established directly peer-to-peer]     │
  sender-3d.py ────── rgb_payload / depth_payload ──────────► receiver-3d.py
                                                              │
                                                         write_video_pyav()
                                                         saves rgb.mp4 + depth.mp4
```

1. **Signaling server** relays SDP offer/answer between sender and receiver.
2. **Sender** reads RGB and depth MP4 files, compresses each frame with the
   chosen codec, and transmits both streams over two unreliable WebRTC
   DataChannels.
3. **Receiver** assembles incoming chunks into frames, decodes them, and writes
   the output to two MP4 files.

### Key design choices

| Design | Detail |
|--------|--------|
| Transport | WebRTC DataChannels — unreliable, unordered (no SCTP retransmit) |
| I-frame protection | Reed-Solomon FEC (zfec k-of-n, ≈50% parity overhead) |
| P-frame recovery | Best-effort: missing chunks are zero-padded |
| Chunk interleaving | RGB and depth chunks alternate within each frame to equalize loss impact |
| Send pacing | Chunks spread evenly across the frame's time slot |
| Deadline clock | Receiver starts a wall-clock deadline on the first decoded I-frame |
| Freeze strategy | Lost frames repeat the last good frame |

---

## Quick start

### 1. Install dependencies
```bash
conda create -n revo python=3.12 
```
```bash
conda activate revo
```
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
```bash
pip install tqdm scipy pybind11 pillow pandas matplotlib pyyaml torchmetrics av aiortc aiohttp torchcodec==0.3 numpy opencv-python zfec einops pytorch_msssim timm
```
Install DCVC-RT dependencies:

**Step 1:** Clone the DCVC-RT repository.
```bash
git clone https://github.com/microsoft/DCVC.git
```
**Step 2:** Build and install the C++ extension.
```bash
cd ./DCVC/src/cpp/
pip install --no-build-isolation .
```
**Step 3:** Download checkpoints from https://github.com/microsoft/DCVC


For more details or troubleshooting DCVC-RT installation please refer to official documentation at https://github.com/microsoft/DCVC

NOTE: Ensure the codec wrapper modules (`H265_wrapper.py`, `H264_wrapper.py`,
`DCVCRT_wrapper.py`) are present in both `src/sender/` and `src/receiver/`.
### 2. Start the signaling server

Run this on any machine reachable by both sender and receiver:

```bash
cd src/
python signalling_server.py
```

Listens on `0.0.0.0:8080`.

### 3. Start the receiver

The receiver must be running before the sender initiates the WebRTC handshake.

1. Edit `src/receiver/run_receiver_eval.py` with your machine IPs, codec, and directory paths.
2. On the receiver machine:

```bash
cd src/receiver/
python run_receiver_eval.py
```

### 4. Start the sender

1. Edit `src/sender/run_sender_eval.py` with your machine IPs, codec, and directory paths.
2. Populate `src/sender/trace_map.txt` with `(category, video_stem, trace_path)` entries.
3. On the sender machine:

```bash
cd src/sender/
sudo python run_sender_eval.py
```

See [sender.md](src/sender/README.md) and [receiver.md](src/receiver/README.md)
for full configuration details.

### 5. Generate frame-corruption masks

Once a streaming session completes, the receiver writes a log file per video.
Parse those logs into per-video `.npy` frame masks (required by the loss recovery step):

1. Edit the `LOG_DIR` and `SAVE_DIR` variables at the top of `scripts/generate_lost_frame_map.py`
   to point to the receiver log directory and your desired mask output directory.
2. Run from the repository root:

```bash
python scripts/generate_lost_frame_map.py
```

### 6. Run loss recovery

Download the pre-trained checkpoints first (one-time setup):

```bash
bash scripts/download_checkpoints.sh
```

Then run inference — see [lossrec.md](src/lossrec/README.md) for the full
command reference for both RGB and depth streams.

---

## Supported codecs

| Flag | Codec |
|------|-------|
| `h265` | H.265 / HEVC (default) |
| `h264` | H.264 / AVC |
| `dcvcrt` | DCVC-RT (neural video codec) |

---

## Network emulation

Network emulation runs automatically — `run_sender_eval.py` launches
`run_loss_trace.py` as a subprocess for each run, passing the trace file
resolved from `trace_map.txt`.  No manual invocation is needed.

Trace files are whitespace-separated with columns:

```
<timestamp_s>  <bandwidth_mbps>  <rtt_ms>  <loss_0_to_1>
```

RTT/delay is fixed at 0 ms; only bandwidth and loss columns are applied via
Linux `tc` / `netem`.  `run_sender_eval.py` requires `sudo` for
the `tc` commands.

---

## 🔧 Troubleshooting

See [FAQ.md](FAQ.md) for known issues and workarounds.

---

## 🎓 Citation

If you find **ReVo** useful in your research, please cite:

```bibtex
@misc{aditya2026revocrosslayerreliablevolumetric,
      title={ReVo: A Cross-Layer Reliable Volumetric Videoconferencing System}, 
      author={Ankur Aditya and Diptyaroop Maji and Lingdong Wang and Bhavya Ramakrishna and Ramesh Sitaraman and Prashant Shenoy},
      year={2026},
      eprint={2604.27441},
      archivePrefix={arXiv},
      primaryClass={cs.NI},
      url={https://arxiv.org/abs/2604.27441}, 
}
```

---

## Acknowledgments

This codebase builds on several excellent open-source projects:

- The neural loss recovery module borrows the ViViT backbone from
  [VideoMAE](https://github.com/MCG-NJU/VideoMAE) (Wang et al., NeurIPS 2022).
  We thank the MCG-NJU team for releasing their code.
- The `dcvcrt` codec integration uses
  [DCVC-RT](https://github.com/microsoft/DCVC) from Microsoft Research.
  We thank the DCVC team for open-sourcing their neural video compression framework.
