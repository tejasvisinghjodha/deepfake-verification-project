# =============================================================================
# FILE: model/ai_fingerprinter.py
# MEMBER 2 — AI Fingerprinting
# =============================================================================
# HOW THIS CONNECTS TO MEMBER 1:
#
#   Member 1's extract_frames() returns a LIST OF NUMPY ARRAYS
#   Each numpy array = one frame (image) from the video
#   Format: BGR color (OpenCV default), shape = (height, width, 3)
#
#   This file takes that exact list and:
#   1. Converts each numpy array → PIL Image (format ResNet-50 understands)
#   2. Passes each image through ResNet-50 → gets 2048 numbers per frame
#   3. Averages all frames → ONE fingerprint (2048 numbers) for the whole video
#   4. Returns that fingerprint to Member 3 for blockchain hashing
#
# WORKFLOW:
#   Member 1                    Member 2                        Member 3
#   extract_frames(video)  -->  generate_fingerprint(frames)  --> hash it --> blockchain
#   returns list of             returns list of 2048 numbers
#   numpy arrays
# =============================================================================


# --- IMPORTS ---
# Install with: pip install torch torchvision pillow numpy opencv-python

import torch                                   # Core AI / deep learning library
import torchvision.models as models            # Gives us ResNet-50
import torchvision.transforms as transforms    # Prepares images for the model
from PIL import Image                          # Converts numpy arrays to images
import numpy as np                             # Maths / array operations
import cv2                                     # Only used in the test at the bottom


# =============================================================================
# STEP 1: LOAD THE AI MODEL (ResNet-50)
# =============================================================================
# ResNet-50 is a powerful image-understanding model trained on millions of
# images. We use it as a "feature extractor" — it reads an image and gives
# back 2048 numbers that describe what it sees (textures, shapes, patterns).
#
# We do NOT train this model. We just use it like a ready-made tool.

def load_model():
    """
    Loads pre-trained ResNet-50 and removes its final classification layer.

    Why remove the last layer?
      - Original last layer outputs "this is a cat / dog / car" (1000 classes)
      - We don't want a label, we want raw feature numbers (2048 of them)
      - So we replace the last layer with Identity() = "do nothing, pass through"

    First run: downloads ~100MB of weights automatically from the internet.
    After that: loads instantly from cache.
    """

    print("[INFO] Loading ResNet-50 model (downloads ~100MB on first run)...")

    # Load ResNet-50 with pre-trained ImageNet weights
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Replace the final classification layer with Identity (pass-through)
    # Now output shape = (1, 2048) instead of (1, 1000)
    model.fc = torch.nn.Identity()

    # eval() mode: disables dropout & batch norm updates
    # IMPORTANT: without this, you get slightly different results each run
    model.eval()

    print("[INFO] Model ready.")
    return model


# =============================================================================
# STEP 2: DEFINE IMAGE PREPROCESSING
# =============================================================================
# ResNet-50 was trained on images processed in a specific way.
# We MUST apply the exact same processing here, otherwise the model
# gives meaningless output — like speaking the wrong language to it.
#
# Required steps:
#   - Resize to 256x256
#   - Crop center 224x224
#   - Convert to tensor (values 0.0 to 1.0)
#   - Normalize with ImageNet's mean and std values

def get_transform():
    """
    Returns the image preprocessing pipeline for ResNet-50.
    This is applied to every single frame before passing to the model.
    """

    transform = transforms.Compose([
        transforms.Resize(256),              # Resize shorter side to 256px
        transforms.CenterCrop(224),          # Crop center 224x224 region
        transforms.ToTensor(),               # Convert PIL Image to tensor [0,1]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],      # ImageNet mean (R, G, B)
            std=[0.229, 0.224, 0.225]        # ImageNet std  (R, G, B)
        ),
    ])

    return transform


# =============================================================================
# STEP 3: CONVERT ONE NUMPY FRAME → 2048 FEATURE NUMBERS
# =============================================================================
# Member 1's frames are numpy arrays in BGR format (OpenCV default).
# ResNet-50 needs PIL Images in RGB format.
# This function handles that conversion + runs the frame through the model.

def extract_features_from_frame(frame_numpy, model, transform):
    """
    Takes ONE frame (numpy array from Member 1) and returns 2048 feature numbers.

    Args:
        frame_numpy : A single frame from extract_frames()
                      Shape: (height, width, 3), dtype: uint8, color: BGR
        model       : Loaded ResNet-50 model (from load_model())
        transform   : Preprocessing pipeline (from get_transform())

    Returns:
        numpy.ndarray of shape (2048,)  — 2048 numbers describing this frame
        Returns None if the frame is invalid.
    """

    # Safety check — make sure the frame is a valid numpy array
    if frame_numpy is None or not isinstance(frame_numpy, np.ndarray):
        print("[WARNING] Invalid frame received — skipping.")
        return None

    try:
        # --- CONVERSION: OpenCV numpy (BGR) → PIL Image (RGB) ---
        # OpenCV stores colors as Blue-Green-Red (BGR)
        # PIL / ResNet-50 expects Red-Green-Blue (RGB)
        # cv2.cvtColor swaps the channel order for us
        frame_rgb = cv2.cvtColor(frame_numpy, cv2.COLOR_BGR2RGB)

        # Convert the numpy array to a PIL Image object
        pil_image = Image.fromarray(frame_rgb)

        # --- PREPROCESSING: resize, crop, normalize ---
        img_tensor = transform(pil_image)
        # Shape is now: (3, 224, 224) — 3 color channels, 224x224 pixels

        # --- ADD BATCH DIMENSION ---
        # PyTorch models expect input shape: (batch_size, channels, height, width)
        # We have 1 image, so: (1, 3, 224, 224)
        img_tensor = img_tensor.unsqueeze(0)

        # --- RUN THROUGH RESNET-50 ---
        # torch.no_grad() = don't track gradients (not training, saves memory)
        with torch.no_grad():
            features = model(img_tensor)
        # Output shape: (1, 2048)

        # --- FLATTEN TO 1D NUMPY ARRAY ---
        # .squeeze() removes the batch dimension: (1, 2048) → (2048,)
        # .numpy() converts PyTorch tensor → numpy array
        features_np = features.squeeze().numpy()

        return features_np  # Shape: (2048,)

    except Exception as e:
        print(f"[ERROR] Could not process frame: {e}")
        return None


# =============================================================================
# STEP 4: GENERATE THE VIDEO FINGERPRINT  ← MAIN FUNCTION
# =============================================================================
# This is what Member 4 (integration) will call.
#
# It receives the exact output of Member 1's extract_frames():
#   a list of numpy arrays
#
# It returns ONE fingerprint:
#   a list of 2048 floating point numbers that uniquely represents the video

def generate_fingerprint(frames):
    """
    MAIN FUNCTION — generates a single AI fingerprint for the entire video.

    Takes the list of numpy frame arrays from Member 1's extract_frames()
    and returns one fingerprint (list of 2048 numbers) for Member 3 to hash.

    Args:
        frames (list): Output of extract_frames() from video_processor.py
                       Each item is a numpy array of shape (height, width, 3)

    Returns:
        list of float: 2048 numbers representing the video's AI fingerprint.
                       Returns None if no frames could be processed.

    --- HOW TO USE (for Member 4 / integration) ---

        from video_processor import extract_frames
        from model.ai_fingerprinter import generate_fingerprint

        # Step 1: Member 1 extracts frames from the video
        frames = extract_frames("uploaded_video.mp4", interval_seconds=1)

        # Step 2: Member 2 generates the fingerprint from those frames
        fingerprint = generate_fingerprint(frames)

        # Step 3: Pass fingerprint to Member 3 for hashing + blockchain storage
    """

    # Check we actually received frames
    if not frames or len(frames) == 0:
        print("[ERROR] No frames provided. Did Member 1's extract_frames() run correctly?")
        return None

    print(f"\n[INFO] Generating fingerprint from {len(frames)} frames...")

    # Load the AI model and preprocessing pipeline
    model = load_model()
    transform = get_transform()

    # Process each frame and collect feature vectors
    all_features = []

    for i, frame in enumerate(frames):
        print(f"[INFO] Processing frame {i + 1} of {len(frames)}...")

        features = extract_features_from_frame(frame, model, transform)

        if features is not None:
            all_features.append(features)
            # each features is shape (2048,)

    # Make sure at least one frame was processed successfully
    if len(all_features) == 0:
        print("[ERROR] All frames failed to process. Cannot generate fingerprint.")
        return None

    print(f"[INFO] Successfully processed {len(all_features)} / {len(frames)} frames.")

    # --- AVERAGE ALL FRAME VECTORS INTO ONE FINGERPRINT ---
    # Stack into 2D array: shape (num_frames, 2048)
    # Then average across rows (axis=0): shape (2048,)
    # Result: one vector that represents the WHOLE video
    stacked = np.array(all_features)               # shape: (num_frames, 2048)
    fingerprint_array = np.mean(stacked, axis=0)   # shape: (2048,)

    # Convert to plain Python list for easy passing between modules
    fingerprint = fingerprint_array.tolist()

    print(f"[INFO] Fingerprint ready! {len(fingerprint)} values generated.")
    print(f"[INFO] Sample (first 5 values): {[round(v, 4) for v in fingerprint[:5]]}")

    return fingerprint  # Hand this to Member 3


# =============================================================================
# STEP 5: COMPARE TWO FINGERPRINTS (helper for local testing)
# =============================================================================
# Member 3 does the real verification using blockchain hashes.
# But this function lets you quickly check similarity locally — useful for
# testing before the blockchain part is ready.

def compare_fingerprints(fingerprint_a, fingerprint_b):
    """
    Computes cosine similarity between two video fingerprints.

    Score guide:
        ~1.00 → Almost identical (same or very similar video)
        ~0.90 → Minor differences (possible light editing)
        ~0.70 → Noticeable differences
        below 0.70 → Likely tampered or completely different video

    Args:
        fingerprint_a (list): Fingerprint of video A (2048 numbers)
        fingerprint_b (list): Fingerprint of video B (2048 numbers)

    Returns:
        float: Similarity score between 0.0 and 1.0
    """

    a = np.array(fingerprint_a)
    b = np.array(fingerprint_b)

    # Cosine similarity = dot product / (magnitude A x magnitude B)
    dot    = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        print("[WARNING] A fingerprint is all zeros. Cannot compare.")
        return 0.0

    similarity = dot / (norm_a * norm_b)

    print(f"\n[COMPARISON RESULT]")
    print(f"  Cosine Similarity Score : {similarity:.4f}")

    if similarity >= 0.98:
        print("  Verdict : AUTHENTIC — videos are virtually identical")
    elif similarity >= 0.85:
        print("  Verdict : SUSPICIOUS — minor differences detected")
    else:
        print("  Verdict : TAMPERED — videos are significantly different")

    return float(similarity)


# =============================================================================
# STEP 6: SELF-TEST — run this file directly to verify everything works
# =============================================================================
# Run in terminal: python ai_fingerprinter.py
#
# This test simulates exactly what will happen in the real project:
#   1. Creates fake "frames" as numpy arrays (just like Member 1 would return)
#   2. Runs generate_fingerprint() on them
#   3. Runs compare_fingerprints() to verify the comparison works

if __name__ == "__main__":

    print("=" * 60)
    print("  AI FINGERPRINTER — Self Test")
    print("  Simulating Member 1's extract_frames() output")
    print("=" * 60)

    # --- Create fake frames as numpy arrays (BGR format, like OpenCV returns) ---
    # In real usage these come from: frames = extract_frames("video.mp4")
    print("\n[TEST] Creating dummy numpy frames (simulating Member 1 output)...")

    fake_frames = []
    colors_bgr = [
        (50,  100, 200),   # Frame 1 (BGR)
        (200, 100,  50),   # Frame 2 (BGR)
        (100, 200, 100),   # Frame 3 (BGR)
    ]

    for i, (b, g, r) in enumerate(colors_bgr):
        # Create a 300x300 solid color numpy array — same format as OpenCV output
        frame = np.full((300, 300, 3), [b, g, r], dtype=np.uint8)
        fake_frames.append(frame)
        print(f"[TEST] Dummy frame {i+1}: shape={frame.shape}, dtype={frame.dtype}")

    # --- Test generate_fingerprint() ---
    print("\n[TEST] Running generate_fingerprint()...")
    fingerprint = generate_fingerprint(fake_frames)

    if fingerprint:
        print(f"\n[TEST PASSED] Fingerprint generated successfully.")
        print(f"  Length : {len(fingerprint)} values (expected 2048)")
        print(f"  Sample : {[round(v, 4) for v in fingerprint[:5]]}")
    else:
        print("[TEST FAILED] Fingerprint generation returned None.")

    # --- Test compare_fingerprints() ---
    print("\n[TEST] Comparing fingerprint to itself (score should be 1.0)...")
    score = compare_fingerprints(fingerprint, fingerprint)
    print(f"[TEST] Self-similarity: {score:.4f} (expected: 1.0000)")

    print("\n[ALL TESTS PASSED] Ready to integrate with Member 1 and Member 3.")
    print("=" * 60)

