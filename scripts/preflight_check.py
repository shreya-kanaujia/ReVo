import sys
import os
import platform
import shutil
import subprocess

PROJECT_ROOT = "/app"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src/sender"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src/receiver"))

def check_step(name, check_fn):
    print(f"Checking {name}...", end="", flush=True)
    try:
        res = check_fn()
        print(f" [PASS] ({res})")
        return True
    except Exception as e:
        print(f" [FAIL] ({e})")
        return False

def check_python_arch():
    return f"{platform.machine()}, {platform.architecture()[0]}"

def check_torch():
    import torch
    return f"version {torch.__version__}"

def check_torchcodec():
    import torchcodec
    return f"version {torchcodec.__version__}"

def check_short_rgb():
    from torchcodec.decoders import VideoDecoder
    path = os.path.join(PROJECT_ROOT, "data/short_rgb.mp4")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    decoder = VideoDecoder(path)
    return f"opened successfully, frames={len(decoder)}, fps={decoder.metadata.average_fps}"

def check_short_depth():
    from torchcodec.decoders import VideoDecoder
    path = os.path.join(PROJECT_ROOT, "data/short_depth.mp4")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    decoder = VideoDecoder(path)
    return f"opened successfully, frames={len(decoder)}, fps={decoder.metadata.average_fps}"

def check_trace_parses():
    path = os.path.join(PROJECT_ROOT, "src/sender/traces/constant_loss_10pct_360s.log")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Invalid trace entry line: {line}")
            time_val = float(parts[0])
            bw_val = float(parts[1])
            rtt_val = int(parts[2])
            loss_val = float(parts[3])
            entries.append((time_val, bw_val, rtt_val, loss_val))
    return f"parsed successfully, entries={len(entries)}"

def check_output_writable():
    path = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(path, exist_ok=True)
    test_file = os.path.join(path, ".writable_test")
    with open(test_file, "w") as f:
        f.write("test")
    os.remove(test_file)
    return "directory is writable"

def check_salsify_mode():
    mode = os.environ.get("SALSIFY_MODE")
    if mode is None:
        return "SALSIFY_MODE not set (defaulting to 0/False)"
    if mode != "0":
        raise ValueError(f"SALSIFY_MODE is set to '{mode}', expected '0'")
    return "SALSIFY_MODE is set to '0' (verified)"

def check_docker_assumptions():
    is_root = os.getuid() == 0
    has_eth0 = os.path.exists("/sys/class/net/eth0")
    return f"running as root={is_root}, eth0 interface exists={has_eth0}"

def check_import_sender():
    import importlib.util
    spec = importlib.util.spec_from_file_location("sender_3d", os.path.join(PROJECT_ROOT, "src/sender/sender-3d.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return "imported sender-3d module successfully"

def check_import_receiver():
    import importlib.util
    spec = importlib.util.spec_from_file_location("receiver_3d", os.path.join(PROJECT_ROOT, "src/receiver/receiver-3d.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return "imported receiver-3d module successfully"

def check_pyav():
    import av
    return f"version {av.__version__}"

def check_libx265():
    import av
    codec = av.Codec('libx265', 'w')
    return f"codec {codec.name} initialized successfully"

def check_hevc_decoder():
    import av
    codec = av.Codec('hevc', 'r')
    return f"codec {codec.name} initialized successfully"

def check_zfec():
    import zfec
    return f"version {zfec.__version__}"

def check_aiortc():
    import aiortc
    return "imported successfully"

def check_aiohttp():
    import aiohttp
    return "imported successfully"

def check_tc_command():
    path = shutil.which("tc")
    if not path:
        raise FileNotFoundError("tc command not found on path")
    res = subprocess.run(["tc", "-V"], capture_output=True, text=True, check=True)
    return f"found at {path}, version: {res.stdout.strip()}"

def main():
    print("=== STARTING REVO PREFLIGHT DEPENDENCY CHECK ===")
    steps = [
        ("Python Architecture", check_python_arch),
        ("Torch Library", check_torch),
        ("Torchcodec Library", check_torchcodec),
        ("Short RGB Video File Check", check_short_rgb),
        ("Short Depth Video File Check", check_short_depth),
        ("Trace File Parsing Check", check_trace_parses),
        ("Output Folder Writable Check", check_output_writable),
        ("Salsify Mode Check", check_salsify_mode),
        ("Docker Context Assumptions", check_docker_assumptions),
        ("Import Sender Module", check_import_sender),
        ("Import Receiver Module", check_import_receiver),
        ("PyAV Library", check_pyav),
        ("libx265 Encoder Creation", check_libx265),
        ("HEVC Decoder Creation", check_hevc_decoder),
        ("zfec Library", check_zfec),
        ("aiortc Library", check_aiortc),
        ("aiohttp Library", check_aiohttp),
        ("Linux tc Command Availability", check_tc_command),
    ]

    all_passed = True
    for name, fn in steps:
        success = check_step(name, fn)
        if not success:
            all_passed = False

    print("================================================")
    if all_passed:
        print("RESULT: ALL PREFLIGHT CHECKS PASSED!")
        sys.exit(0)
    else:
        print("RESULT: SOME PREFLIGHT CHECKS FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()
