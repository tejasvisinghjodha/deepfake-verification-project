"""
Member 2 — AI Fingerprint Generation
======================================
Uses a pretrained ResNet-50 CNN (ImageNet weights) as a feature extractor.
The final classification layer is removed — we use the 2048-dimensional
penultimate layer as the video's visual "DNA".

Process:
  1. Each frame is normalized and batched
  2. ResNet-50 extracts a 2048-d feature vector per frame
  3. Vectors are averaged across all frames → one fingerprint per video
  4. The fingerprint is a Python list of 2048 floats (JSON-serialisable)

Why ResNet-50?
  - Strong visual features without training a deepfake detector from scratch
  - Any edit (crop, color grading, watermark, face swap) changes the features
  - Fast inference on CPU for prototyping; GPU optional

Optional deepfake detection:
  - See `deepfake_score()` method — requires a fine-tuned binary classifier
    head, which you can add in Week 2 using FaceForensics++ data.

Usage:
    from model.ai_fingerprinter import AIFingerprinter
    fingerprinter = AIFingerprinter()
    vector = fingerprinter.fingerprint(frames)  # frames from VideoProcessor
"""

import numpy as np
from typing import List


class AIFingerprinter:
    """
    Generates a unique AI fingerprint from a list of video frames.

    Parameters
    ----------
    device : str
        'cuda' if GPU available, otherwise 'cpu'. Auto-detected if None.
    batch_size : int
        Number of frames to process in one forward pass.
    """

    def __init__(self, device: str = None, batch_size: int = 16):
        import torch
        import torchvision.models as models
        import torchvision.transforms as transforms

        # Auto-detect GPU
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.batch_size = batch_size
        self.torch = torch

        # -----------------------------------------------------------------
        # Load pretrained ResNet-50, strip the classification head
        # We keep everything up to the global average pooling layer,
        # which outputs a 2048-d feature vector per image.
        # -----------------------------------------------------------------
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Replace final FC layer with Identity to get raw 2048-d features
        base_model.fc = torch.nn.Identity()

        self.model = base_model.to(self.device)
        self.model.eval()  # Inference mode — no gradient tracking needed

        # ImageNet normalisation (required for pretrained ResNet weights)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        print(f"[AIFingerprinter] Loaded ResNet-50 on {self.device}")

    def fingerprint(self, frames: List[np.ndarray]) -> List[float]:
        """
        Generate a single fingerprint vector from a list of frames.

        Parameters
        ----------
        frames : list of np.ndarray
            RGB frames, each (224, 224, 3) uint8 — output of VideoProcessor.

        Returns
        -------
        list of float
            2048-dimensional fingerprint vector, averaged across all frames.
            JSON-serialisable — safe to pass to SHA-256 hashing.
        """
        import torch
        from PIL import Image

        if not frames:
            raise ValueError("No frames provided to fingerprint()")

        all_features = []

        # Process in batches to avoid OOM on large frame sets
        for i in range(0, len(frames), self.batch_size):
            batch_frames = frames[i : i + self.batch_size]

            # Convert numpy arrays → PIL Images → tensors
            tensors = [self.transform(Image.fromarray(f)) for f in batch_frames]
            batch_tensor = torch.stack(tensors).to(self.device)

            with torch.no_grad():
                features = self.model(batch_tensor)  # (B, 2048)

            # Move to CPU and convert to numpy
            all_features.append(features.cpu().numpy())

        # Stack all batch outputs: shape (N_frames, 2048)
        feature_matrix = np.vstack(all_features)

        # Average across frames → single 2048-d fingerprint
        fingerprint_vector = feature_matrix.mean(axis=0)

        # Normalise to unit length for consistent hashing behaviour
        norm = np.linalg.norm(fingerprint_vector)
        if norm > 0:
            fingerprint_vector = fingerprint_vector / norm

        # Return as plain Python list for JSON serialisation
        return fingerprint_vector.tolist()

    def deepfake_score(self, frames: List[np.ndarray]) -> float:
        """
        Optional: returns a 0–1 probability that the video is AI-generated.

        NOTE: This requires a fine-tuned binary classifier head.
        For the prototype, this returns a placeholder.
        In Week 2, load a classifier trained on FaceForensics++ here.

        Returns
        -------
        float
            0.0 = likely authentic, 1.0 = likely deepfake.
        """
        # TODO (Member 2, Week 2):
        #   1. Load a saved binary classifier: self.classifier = torch.load('classifier.pt')
        #   2. Run features through it: score = self.classifier(features).sigmoid()
        #   3. Return the float score
        raise NotImplementedError(
            "Deepfake classifier not yet trained. "
            "Implement in Week 2 using FaceForensics++ dataset."
        )
