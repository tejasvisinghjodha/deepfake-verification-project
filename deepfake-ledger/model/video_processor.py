"""
Member 1 — Video Processing & Frame Extraction
================================================
Handles:
  - Reading video files with OpenCV
  - Extracting a representative sample of frames
  - Pulling video metadata (resolution, fps, duration, codec)

Usage:
    from model.video_processor import VideoProcessor
    processor = VideoProcessor(frame_interval=10)
    frames, metadata = processor.extract("path/to/video.mp4")
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple


class VideoProcessor:
    """
    Extracts frames and metadata from a video file.

    Parameters
    ----------
    frame_interval : int
        Extract every Nth frame. Default=10 gives ~3 fps for a 30fps video,
        which is sufficient for fingerprinting and keeps processing fast.
    max_frames : int
        Hard cap on the number of frames extracted. Prevents memory issues
        on very long videos. Default=100 frames.
    target_size : tuple
        Resize each frame to (width, height) before returning.
        Smaller frames = faster feature extraction downstream.
    """

    def __init__(
        self,
        frame_interval: int = 10,
        max_frames: int = 100,
        target_size: Tuple[int, int] = (224, 224),
    ):
        self.frame_interval = frame_interval
        self.max_frames = max_frames
        self.target_size = target_size

    def extract(self, video_path: str) -> Tuple[list, dict]:
        """
        Open a video and extract frames + metadata.

        Parameters
        ----------
        video_path : str
            Path to the video file.

        Returns
        -------
        frames : list of np.ndarray
            List of RGB frames, each shaped (H, W, 3), dtype uint8.
        metadata : dict
            Video properties: fps, frame_count, duration_seconds,
            width, height, codec.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open: {video_path}")

        # --- Collect metadata ---
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        # Four-character codec code
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

        metadata = {
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration_seconds": round(duration, 2),
            "width": width,
            "height": height,
            "codec": codec.strip(),
            "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        }

        # --- Extract frames ---
        frames = []
        frame_idx = 0

        while cap.isOpened() and len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Sample every Nth frame
            if frame_idx % self.frame_interval == 0:
                # OpenCV reads BGR — convert to RGB for PyTorch/PIL compatibility
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize to target_size for uniform input to the AI model
                frame_resized = cv2.resize(
                    frame_rgb,
                    self.target_size,
                    interpolation=cv2.INTER_AREA,
                )
                frames.append(frame_resized)

            frame_idx += 1

        cap.release()

        if len(frames) == 0:
            raise ValueError("Could not extract any frames from the video.")

        return frames, metadata
