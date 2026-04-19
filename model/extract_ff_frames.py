"""
Member 2 — FaceForensics++ Frame Extractor
===========================================
Run this script to prepare training data from the FaceForensics++ dataset.

Usage:
    python model/extract_ff_frames.py

Before running:
    1. Download FaceForensics++ from https://github.com/ondyari/FaceForensics
       (requires filling a Google Form for academic access — takes ~1 day)
    2. Put the downloaded videos in:
         data/ff_videos/real/    ← original (real) videos
         data/ff_videos/fake/    ← manipulated videos (DeepFakes folder)
    3. Run this script — it will extract frames into data/real/ and data/fake/
    4. Then run: python model/deepfake_detector.py --train
"""

import cv2
import os
from pathlib import Path

# How many frames to extract per video
FRAMES_PER_VIDEO = 30

# Input video folders
REAL_VIDEO_DIR = "data/ff_videos/real"
FAKE_VIDEO_DIR = "data/ff_videos/fake"

# Output frame folders
REAL_FRAME_DIR = "data/real"
FAKE_FRAME_DIR = "data/fake"


def extract_frames(video_path: str, output_dir: str, max_frames: int = 30):
    """Extract evenly-spaced frames from a video."""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total // max_frames)

    video_name = Path(video_path).stem
    count = 0
    frame_idx = 0

    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (224, 224))
            out_path = os.path.join(output_dir, f"{video_name}_frame{count:04d}.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR))
            count += 1
        frame_idx += 1

    cap.release()
    return count


def run():
    for label, video_dir, frame_dir in [
        ("REAL", REAL_VIDEO_DIR, REAL_FRAME_DIR),
        ("FAKE", FAKE_VIDEO_DIR, FAKE_FRAME_DIR),
    ]:
        videos = list(Path(video_dir).rglob("*.mp4"))
        videos += list(Path(video_dir).rglob("*.avi"))
        print(f"\n[{label}] Found {len(videos)} videos in {video_dir}")

        total_frames = 0
        for i, vpath in enumerate(videos):
            n = extract_frames(str(vpath), frame_dir, FRAMES_PER_VIDEO)
            total_frames += n
            print(f"  [{i+1}/{len(videos)}] {vpath.name} → {n} frames")

        print(f"[{label}] Done. Total frames extracted: {total_frames}")

    print("\nAll done! Now run: python model/deepfake_detector.py --train")


if __name__ == "__main__":
    run()
