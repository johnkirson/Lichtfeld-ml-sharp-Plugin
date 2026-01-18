#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHARP 4D Video Processor using the SHARP CLI (In-Process).
"""

import sys
import os
import shutil
import tempfile
from pathlib import Path
import logging
import numpy as np
import imageio.v2 as imageio
import imageio_ffmpeg
from tqdm import tqdm
from unittest.mock import patch

# Ensure ml-sharp is in path
_THIS_DIR = Path(__file__).parent.resolve()
_ML_SHARP_SRC = _THIS_DIR / "ml-sharp" / "src"
if str(_ML_SHARP_SRC) not in sys.path:
    sys.path.insert(0, str(_ML_SHARP_SRC))

# Import the CLI command directly
from sharp.cli.predict import predict_cli
from sharp.utils import logging as sharp_logging
from sharp.utils.gaussians import load_ply
from plyfile import PlyData

# Force imageio to use the ffmpeg binary from the imageio-ffmpeg package
os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

class SharpProcessor:
    def __init__(self):
        # We don't want to reset basicConfig if LFS has set it up, but we get a logger
        self.logger = logging.getLogger("SharpProcessor")

    def process_video(self, video_path: str, output_dir: str, progress_callback=None) -> tuple[list[str], float]:
        """
        Process a video file using the 'sharp predict' CLI command (in-process).
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Create temporary directory for frames
        temp_dir = Path(tempfile.mkdtemp(prefix="sharp_frames_"))
        try:
            self.logger.info(f"Extracting frames to {temp_dir}")
            
            # Ensure we are passing a string
            video_path_str = str(video_path)
            
            # Force ffmpeg backend to ensure MP4 support
            reader = imageio.get_reader(video_path_str, format='ffmpeg')
            meta = reader.get_meta_data()
            fps = meta.get("fps", 30.0)
            
            try:
                total_frames = reader.count_frames()
            except:
                total_frames = 0

            for i, frame in enumerate(reader):
                if progress_callback:
                    progress_callback(i, total_frames, f"Extracting frame {i+1}")
                
                frame_path = temp_dir / f"frame_{i:05d}.jpg"
                imageio.imsave(frame_path, frame)
            reader.close()

            # 2. Run sharp predict CLI (In-Process)
            self.logger.info("Running sharp predict CLI...")
            if progress_callback:
                progress_callback(0, 1, "Running SHARP Inference (this may take a while)...")

            # Prepare arguments for click
            # sharp predict -i <temp_dir> -o <output_dir>
            # Note: predict_cli is the command function decorated with @click.command
            
            args = [
                "-i", str(temp_dir),
                "-o", str(output_dir),
                "--device", "cuda" # explicit device preference
            ]

            # Patch sharp.utils.logging.configure to avoid messing up LFS logging
            # We just mock it to do nothing, or we could redirect it.
            with patch('sharp.utils.logging.configure') as mock_log_conf:
                try:
                    # standalone_mode=False prevents click from calling sys.exit()
                    predict_cli.main(args=args, standalone_mode=False, prog_name="sharp predict")
                except SystemExit as e:
                    if e.code != 0:
                        raise RuntimeError(f"SHARP CLI exited with code {e.code}")
                except Exception as e:
                    self.logger.error(f"SHARP CLI failed: {e}")
                    raise RuntimeError(f"SHARP CLI Error: {e}")

            self.logger.info("SHARP Inference complete.")
            
            # 3. Collect generated PLY files
            ply_files = sorted([str(p) for p in output_dir.glob("frame_*.ply")])
            return ply_files, fps

        finally:
            # Cleanup temp frames
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

def load_gaussian_ply(ply_path):
    """
    Load a Gaussian Splat PLY file and return tensors suitable for scene.add_splat()

    Returns:
        means    : [N, 3]
        sh0      : [N, 1, 3]
        scaling  : [N, 3]
        rotation : [N, 4]  (wxyz)
        opacity  : [N, 1]
    """
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data

    # --- Means ---
    means = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    # --- SH0 (RGB) ---
    sh0 = np.stack(
        [v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]],
        axis=1
    ).astype(np.float32)
    sh0 = sh0[:, None, :]  # [N, 1, 3]

    # --- Opacity ---
    opacity = v["opacity"].astype(np.float32)[:, None]

    # --- Scaling ---
    scaling = np.stack(
        [v["scale_0"], v["scale_1"], v["scale_2"]],
        axis=1
    ).astype(np.float32)

    # --- Rotation ---
    rotation = np.stack(
        [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
        axis=1
    ).astype(np.float32)

    # IMPORTANT: ensure wxyz order
    # If SHARP writes xyzw, swap here:
    # rotation = rotation[:, [3, 0, 1, 2]]

    return means, sh0, scaling, rotation, opacity
def extract_data_from_ply(ply_path):
    """
    Extract point cloud data (means and colors) from a SHARP PLY file.
    """
    gaussians, metadata = load_ply(Path(ply_path))
    xyz = gaussians.mean_vectors.detach().cpu().numpy().reshape(-1, 3)
    rgb = gaussians.colors.detach().cpu().numpy().reshape(-1, 3)
    rgb = np.clip(rgb, 0.0, 1.0)
    return xyz, rgb

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Input video path")
    parser.add_argument("output", help="Output directory")
    args = parser.parse_args()
    
    proc = SharpProcessor()
    files, fps = proc.process_video(args.video, args.output, lambda i, t, m: print(f"{m} ({i}/{t})"))
    print(f"Processed {len(files)} frames at {fps} FPS.")
