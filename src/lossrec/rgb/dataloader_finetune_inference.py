"""
Dataset for RGB video packet loss recovery inference.

Each sample is a clip of `num_frames` frames:
    [clean_{t-k}, ..., clean_{t-1}, corrupted_t]

For every corrupted frame in a video (mask value > 0), a clip is built from
its (num_frames - 1) most recent clean predecessors.  Clips that lack enough
clean history are silently discarded at construction time.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms_v2
from torchcodec.decoders import VideoDecoder


class LossRecoveryData(Dataset):
    """PyTorch Dataset for packet-loss recovery inference (RGB).

    Args:
        clean_dir:     Directory of ground-truth videos (.mp4 / .avi).
        corrupted_dir: Directory of codec-degraded videos (same filenames as clean).
        mask_dir:      Directory of per-video frame masks (<stem>_frame_mask.npy).
                       Mask value 0 = clean frame, > 0 = corrupted frame.
        num_frames:    Total frames per clip, including the corrupted target frame.
        input_size:    Spatial resolution to resize frames to (square, in pixels).
    """

    def __init__(self, clean_dir, corrupted_dir, mask_dir, num_frames=16, input_size=224):
        self.clean_dir = clean_dir
        self.corrupted_dir = corrupted_dir
        self.mask_dir = mask_dir
        self.num_frames = num_frames
        self.input_size = input_size

        self.transform = transforms_v2.Compose([
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Resize(size=(input_size, input_size), antialias=True),
        ])

        self.clips = []      # list of (clean_path, corrupted_path, mask_path, frame_indices)
        self.mask_cache = {} # avoids re-loading the same .npy mask from disk repeatedly

        loaded, skipped = [], []
        total_corrupted = 0
        total_clips = 0

        video_files = [f for f in os.listdir(clean_dir) if f.endswith((".mp4", ".avi"))]
        print("Scanning videos to build clip index...")

        for video_filename in video_files:
            video_name     = os.path.splitext(video_filename)[0]
            clean_path     = os.path.join(clean_dir, video_filename)
            corrupted_path = os.path.join(corrupted_dir, video_filename)
            mask_path      = os.path.join(mask_dir, video_name + "_frame_mask.npy")

            if not (os.path.exists(clean_path) and
                    os.path.exists(corrupted_path) and
                    os.path.exists(mask_path)):
                skipped.append(f"{video_name} (missing file)")
                continue

            try:
                len_clean = len(VideoDecoder(clean_path, device="cpu"))
                len_corr  = len(VideoDecoder(corrupted_path, device="cpu"))

                if mask_path not in self.mask_cache:
                    self.mask_cache[mask_path] = np.load(mask_path, allow_pickle=True)
                raw_mask = self.mask_cache[mask_path]

                # Align the three sources before trimming.
                #
                # The receiver starts writing its output at the first keyframe it
                # can decode, which is not necessarily fid 0 (commonly fid 30).
                # The clean video and the mask are both indexed from fid 0, so
                # indexing all three by the same i compares a ground-truth frame
                # against a received frame a full GOP later, AND reads the mask
                # entry for the wrong frame -- so the model reconstructs frames
                # that were fine and skips the ones that were damaged.
                #
                # LOSSREC_FID0 is the fid the corrupted video starts at; take it
                # from the receiver log rather than assuming zero.
                fid0 = int(os.environ.get("LOSSREC_FID0", "0"))
                clean_off = raw_mask[fid0:] if fid0 else raw_mask
                valid_len  = min(len_clean - fid0, len_corr, len(clean_off))
                frame_mask = clean_off[:valid_len]
                self.clean_offset = fid0

                total_corrupted += int(np.count_nonzero(frame_mask))

                corrupted_indices = np.where(frame_mask > 0)[0]
                clean_indices     = np.where(frame_mask == 0)[0]

                clips_before = len(self.clips)
                for c_idx in corrupted_indices:
                    # Collect the (num_frames - 1) most recent clean frames before c_idx
                    predecessors = clean_indices[clean_indices < c_idx]
                    if len(predecessors) >= self.num_frames - 1:
                        context = predecessors[-(self.num_frames - 1):]
                        indices = list(np.concatenate((context, [c_idx])))
                        self.clips.append((clean_path, corrupted_path, mask_path, indices))

                clips_added = len(self.clips) - clips_before
                total_clips += clips_added
                if clips_added > 0:
                    loaded.append(video_name)
                else:
                    skipped.append(f"{video_name} (0 clips — insufficient clean history)")

            except Exception as e:
                skipped.append(f"{video_name} (error: {e})")

        print(f"\n{'='*50}")
        print(f"Videos scanned:              {len(video_files)}")
        print(f"Videos loaded:               {len(loaded)}")
        print(f"Videos skipped:              {len(skipped)}")
        print(f"Corrupted frames (trimmed):  {total_corrupted}")
        print(f"Clips created:               {total_clips}")
        print(f"Frames dropped (no history): {total_corrupted - total_clips}")
        if skipped:
            print("Skipped:")
            for v in skipped:
                print(f"  - {v}")
        print('=' * 50)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clean_path, corrupted_path, mask_path, frame_indices = self.clips[idx]

        clean_dec = VideoDecoder(clean_path, device="cpu")
        corr_dec  = VideoDecoder(corrupted_path, device="cpu")

        # frame_indices are positions in the CORRUPTED video. The clean video is
        # indexed from fid 0, so reading it at the same position compares frames a
        # GOP apart. Shift the clean read by the receiver's starting fid.
        off = getattr(self, "clean_offset", 0)
        safe_corr  = len(corr_dec) - 1
        safe_clean = len(clean_dec) - 1
        frame_indices  = [min(i, safe_corr) for i in frame_indices]
        clean_indices  = [min(i + off, safe_clean) for i in frame_indices]

        clean_frames     = torch.stack([clean_dec[i] for i in clean_indices])   # [T, C, H, W]
        corrupted_frames = torch.stack([corr_dec[i]  for i in frame_indices])   # [T, C, H, W]

        clean_clip     = self.transform(clean_frames).permute(1, 0, 2, 3)       # [C, T, H, W]
        corrupted_clip = self.transform(corrupted_frames).permute(1, 0, 2, 3)   # [C, T, H, W]

        # self.mask_cache holds the RAW mask (fid-indexed); shift to match.
        clip_mask = torch.from_numpy(
            self.mask_cache[mask_path][[min(i + off, len(self.mask_cache[mask_path]) - 1)
                                        for i in frame_indices]]).long()
        if len(clip_mask) < self.num_frames:
            pad = self.num_frames - len(clip_mask)
            clip_mask = torch.cat([clip_mask, torch.zeros(pad, dtype=torch.long)])

        video_name = os.path.splitext(os.path.basename(clean_path))[0]
        meta = {
            "video_name":  video_name,
            "mask_path":   mask_path,
            "start_frame": frame_indices[self.num_frames - 1],  # global index of the target frame
        }

        # Returns: (corrupted_clip, frame_mask), clean_clip, metadata, frame_indices
        return (corrupted_clip, clip_mask), clean_clip, meta, frame_indices
