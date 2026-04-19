"""
Member 2 — Deepfake Detector
==============================
Detects AI-generated / manipulated faces in video frames WITHOUT
needing a registered video ID. Works on ANY video.

How it works (plain English):
  - We use a pre-trained EfficientNet-B4 model from the `timm` library.
  - It was originally trained on ImageNet (millions of real-world images).
  - We add a binary classification head on top: Real (0) vs Fake (1).
  - We then fine-tune it on deepfake data using FaceForensics++ frames.
  - At inference time, we pass each frame through the model, get a
    probability, and average across all frames for a final score.

If you haven't trained yet:
  - Run this file directly: python model/deepfake_detector.py
  - It will download the base model and run in DEMO MODE using
    random weights — scores won't be accurate but the pipeline works.
  - To get accurate scores, follow the training section below.

Training the model (Member 2, Week 2-3):
  - Download FaceForensics++ dataset (free, academic registration)
  - Put real frames in  data/real/   and fake frames in  data/fake/
  - Run: python model/deepfake_detector.py --train
  - This saves deepfake_model.pt which the detector loads automatically.

Install:
  pip install timm
"""

import os
import numpy as np
from typing import List, Dict
from pathlib import Path


# Path where the trained model weights will be saved/loaded
MODEL_SAVE_PATH = Path("model/deepfake_model.pt")


class DeepfakeDetector:
    """
    Analyses video frames and returns a deepfake probability score.

    Score meaning:
        0.0 – 0.3  →  LOW RISK  — video looks authentic
        0.3 – 0.6  →  MEDIUM RISK — suspicious, needs review
        0.6 – 1.0  →  HIGH RISK — likely deepfake or AI-manipulated

    Parameters
    ----------
    device : str
        'cuda' for GPU (faster), 'cpu' for no GPU. Auto-detected if None.
    confidence_threshold : float
        Score above which we label the video as a deepfake. Default 0.5.
    """

    def __init__(self, device: str = None, confidence_threshold: float = 0.5):
        import torch
        import torch.nn as nn

        self.confidence_threshold = confidence_threshold
        self.torch = torch
        self.nn = nn

        # Auto-detect GPU
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[DeepfakeDetector] Initialising on {self.device}...")

        # Build the model architecture
        self.model = self._build_model()

        # Load trained weights if they exist, otherwise use base weights
        if MODEL_SAVE_PATH.exists():
            print(f"[DeepfakeDetector] Loading trained weights from {MODEL_SAVE_PATH}")
            state = torch.load(MODEL_SAVE_PATH, map_location=self.device)
            self.model.load_state_dict(state)
            self.is_trained = True
        else:
            print("[DeepfakeDetector] No trained weights found — running with base ImageNet weights.")
            print("[DeepfakeDetector] Scores will be approximate until you train on deepfake data.")
            print("[DeepfakeDetector] Run: python model/deepfake_detector.py --train")
            self.is_trained = False

        self.model.eval()

        # Pre-processing: same normalisation EfficientNet expects
        import torchvision.transforms as transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        print(f"[DeepfakeDetector] Ready. Trained={self.is_trained}")

    def _build_model(self):
        """
        Build EfficientNet-B4 with a binary classification head.

        EfficientNet is chosen because:
          - More accurate than ResNet on fine-grained visual artifacts
          - Fast enough to run on CPU for a prototype
          - Pretrained weights available via timm
        """
        import timm
        import torch.nn as nn

        # Load EfficientNet-B4 pretrained on ImageNet
        # num_classes=0 removes the default classification head
        backbone = timm.create_model("efficientnet_b4", pretrained=True, num_classes=0)

        # Get the number of features the backbone outputs
        num_features = backbone.num_features  # typically 1792 for EfficientNet-B4

        # Build our custom classification head
        # Real (class 0) vs Fake (class 1)
        model = nn.Sequential(
            backbone,
            nn.Dropout(p=0.3),           # Dropout reduces overfitting
            nn.Linear(num_features, 256), # Compress features
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),           # Single output: fake probability
            nn.Sigmoid(),                # Squash to 0-1 range
        )

        return model.to(self.device)

    # ------------------------------------------------------------------
    # Main public method — call this from app.py
    # ------------------------------------------------------------------

    def predict(self, frames: List[np.ndarray]) -> Dict:
        """
        Analyse a list of video frames and return a deepfake assessment.

        Parameters
        ----------
        frames : list of np.ndarray
            RGB frames (H, W, 3) uint8 — same format as VideoProcessor output.
            Tip: pass the same frames you used for fingerprinting.

        Returns
        -------
        dict with:
            score        : float 0.0–1.0 (higher = more likely fake)
            verdict      : str  'LIKELY AUTHENTIC' / 'SUSPICIOUS' / 'LIKELY DEEPFAKE'
            risk_level   : str  'LOW' / 'MEDIUM' / 'HIGH'
            confidence   : str  human-readable percentage
            frame_scores : list of per-frame scores (useful for highlighting suspicious moments)
            is_trained   : bool — whether using trained or base weights
        """
        import torch

        if not frames:
            raise ValueError("No frames provided to predict()")

        frame_scores = []

        with torch.no_grad():
            for frame in frames:
                # Preprocess single frame
                tensor = self.transform(frame).unsqueeze(0).to(self.device)

                # Forward pass — get fake probability for this frame
                score = self.model(tensor).item()
                frame_scores.append(score)

        # Average score across all frames
        avg_score = float(np.mean(frame_scores))

        # Determine risk level and verdict
        if avg_score < 0.3:
            risk_level = "LOW"
            verdict = "LIKELY AUTHENTIC"
        elif avg_score < 0.6:
            risk_level = "MEDIUM"
            verdict = "SUSPICIOUS — manual review recommended"
        else:
            risk_level = "HIGH"
            verdict = "LIKELY DEEPFAKE"

        return {
            "score":        round(avg_score, 4),
            "verdict":      verdict,
            "risk_level":   risk_level,
            "confidence":   f"{round(avg_score * 100, 1)}% probability of being fake",
            "frame_scores": [round(s, 4) for s in frame_scores],
            "is_trained":   self.is_trained,
            "frames_analysed": len(frames),
        }

    # ------------------------------------------------------------------
    # Training — run this once when you have FaceForensics++ data
    # ------------------------------------------------------------------

    def train(
        self,
        real_frames_dir: str = "data/real",
        fake_frames_dir: str = "data/fake",
        epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
    ):
        """
        Fine-tune the model on your deepfake dataset.

        Setup before calling:
          1. Download FaceForensics++ from:
             https://github.com/ondyari/FaceForensics
          2. Extract frames from the videos:
             python model/extract_ff_frames.py
          3. Your folder structure should look like:
             data/
               real/   ← frames from real videos (jpg/png)
               fake/   ← frames from deepfake videos (jpg/png)
          4. Then call: python model/deepfake_detector.py --train

        The trained model is saved to model/deepfake_model.pt
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
        from PIL import Image
        import glob

        print("[Training] Starting fine-tuning on deepfake dataset...")
        print(f"[Training] Real frames dir : {real_frames_dir}")
        print(f"[Training] Fake frames dir : {fake_frames_dir}")

        # ---- Dataset ----
        class FrameDataset(Dataset):
            def __init__(self, real_dir, fake_dir, transform):
                real_files = glob.glob(f"{real_dir}/**/*.jpg", recursive=True)
                real_files += glob.glob(f"{real_dir}/**/*.png", recursive=True)
                fake_files = glob.glob(f"{fake_dir}/**/*.jpg", recursive=True)
                fake_files += glob.glob(f"{fake_dir}/**/*.png", recursive=True)

                # 0 = real, 1 = fake
                self.samples = [(f, 0.0) for f in real_files] + \
                               [(f, 1.0) for f in fake_files]
                self.transform = transform
                print(f"[Dataset] {len(real_files)} real + {len(fake_files)} fake frames")

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                path, label = self.samples[idx]
                img = Image.open(path).convert("RGB")
                return self.transform(img), torch.tensor([label], dtype=torch.float32)

        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        dataset = FrameDataset(real_frames_dir, fake_frames_dir, train_transform)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        # ---- Optimiser & loss ----
        # Only fine-tune the classification head + last backbone block
        # This is much faster than training everything from scratch
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCELoss()

        self.model.train()
        best_loss = float("inf")

        for epoch in range(epochs):
            epoch_loss = 0.0
            correct    = 0
            total      = 0

            for batch_frames, labels in loader:
                batch_frames = batch_frames.to(self.device)
                labels       = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_frames)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                preds  = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

            scheduler.step()
            avg_loss = epoch_loss / len(loader)
            accuracy = correct / total * 100

            print(f"[Training] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.1f}%")

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), MODEL_SAVE_PATH)
                print(f"[Training] Saved best model to {MODEL_SAVE_PATH}")

        self.model.eval()
        self.is_trained = True
        print(f"[Training] Done! Model saved to {MODEL_SAVE_PATH}")
        print("[Training] Restart the Flask server to load the trained weights.")


# ------------------------------------------------------------------
# Run directly for testing or training
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if "--train" in sys.argv:
        # Training mode
        print("=" * 50)
        print("TRAINING MODE")
        print("Make sure you have data/real/ and data/fake/ folders")
        print("=" * 50)
        detector = DeepfakeDetector()
        detector.train(
            real_frames_dir="data/real",
            fake_frames_dir="data/fake",
            epochs=10,
        )
    else:
        # Quick test with a random frame (just to check the pipeline works)
        print("=" * 50)
        print("TEST MODE — using random dummy frames")
        print("This just tests the pipeline, not accuracy")
        print("=" * 50)
        detector = DeepfakeDetector()

        # Create 10 random fake frames (224x224 RGB)
        dummy_frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                        for _ in range(10)]

        result = detector.predict(dummy_frames)
        print("\nResult:")
        for key, val in result.items():
            if key != "frame_scores":
                print(f"  {key}: {val}")
        print(f"  frame_scores (first 5): {result['frame_scores'][:5]}")
        print("\nPipeline working correctly!")
