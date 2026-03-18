"""
AI-Driven Deepfake Insurance & Forensic Ledger
================================================
Flask backend — handles video upload, AI fingerprinting,
deepfake detection, SHA-256 hashing, blockchain storage, and verification.

Two-layer verification system:
  Layer 1 — Deepfake Detection  : works on ANY video, no registration needed.
                                   Returns a 0-1 probability of being fake.
  Layer 2 — Hash Verification   : compares blockchain hash with recomputed hash.
                                   Returns AUTHENTIC or TAMPERED.
"""

import os
import json
import hashlib
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Lazy-loaded singletons
_video_processor   = None
_ai_fingerprinter  = None
_deepfake_detector = None
_blockchain_client = None


def get_video_processor():
    global _video_processor
    if _video_processor is None:
        from model.video_processor import VideoProcessor
        _video_processor = VideoProcessor()
    return _video_processor


def get_ai_fingerprinter():
    global _ai_fingerprinter
    if _ai_fingerprinter is None:
        from model.ai_fingerprinter import AIFingerprinter
        _ai_fingerprinter = AIFingerprinter()
    return _ai_fingerprinter


def get_deepfake_detector():
    global _deepfake_detector
    if _deepfake_detector is None:
        from model.deepfake_detector import DeepfakeDetector
        _deepfake_detector = DeepfakeDetector()
    return _deepfake_detector


def get_blockchain_client():
    global _blockchain_client
    if _blockchain_client is None:
        from contracts.blockchain_client import BlockchainClient
        _blockchain_client = BlockchainClient()
    return _blockchain_client


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/register", methods=["POST"])
def register_video():
    """
    Upload a video → extract frames → detect deepfake → fingerprint
    → hash → store on blockchain. Returns full results.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Invalid or unsupported file type"}), 400

    filename   = secure_filename(file.filename)
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(video_path)

    try:
        # Step 1 — Extract frames
        frames, metadata = get_video_processor().extract(video_path)

        # Step 2 — Deepfake detection (NEW — runs on every video)
        deepfake_result = get_deepfake_detector().predict(frames)

        # Step 3 — AI fingerprint
        fingerprint_vector = get_ai_fingerprinter().fingerprint(frames)

        # Step 4 — SHA-256 hash
        video_hash = hashlib.sha256(
            json.dumps(fingerprint_vector).encode()
        ).hexdigest()

        # Step 5 — Store on blockchain
        video_id   = hashlib.md5(filename.encode()).hexdigest()[:12]
        tx_receipt = get_blockchain_client().store_hash(video_id, video_hash)

        return jsonify({
            "success":  True,
            "video_id": video_id,
            "filename": filename,
            "metadata": metadata,
            "deepfake": _format_deepfake(deepfake_result),
            "hash":         video_hash,
            "tx_hash":      tx_receipt["tx_hash"],
            "block_number": tx_receipt["block_number"],
            "explorer_url": tx_receipt.get("explorer_url", ""),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


@app.route("/api/analyse", methods=["POST"])
def analyse_only():
    """
    Run deepfake detection ONLY on any video — no blockchain, no registration needed.
    This is the route for checking viral/suspicious videos that were never registered.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    filename   = secure_filename(file.filename)
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], f"analyse_{filename}")
    file.save(video_path)

    try:
        frames, metadata = get_video_processor().extract(video_path)
        deepfake_result  = get_deepfake_detector().predict(frames)

        return jsonify({
            "success":  True,
            "filename": filename,
            "metadata": metadata,
            "deepfake": _format_deepfake(deepfake_result),
            "note": "This video has not been registered on the blockchain.",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


@app.route("/api/verify", methods=["POST"])
def verify_video():
    """
    Re-upload a video + its video_id. Runs both verification layers:
      Layer 1 — Deepfake AI score
      Layer 2 — Blockchain hash comparison
    Returns a combined verdict.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_id = request.form.get("video_id", "").strip()
    if not video_id:
        return jsonify({"error": "video_id is required"}), 400

    file = request.files["video"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    filename   = secure_filename(file.filename)
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], f"verify_{filename}")
    file.save(video_path)

    try:
        frames, _ = get_video_processor().extract(video_path)

        # Layer 1 — Deepfake detection
        deepfake_result = get_deepfake_detector().predict(frames)

        # Layer 2 — Hash comparison
        fingerprint_vector = get_ai_fingerprinter().fingerprint(frames)
        new_hash = hashlib.sha256(
            json.dumps(fingerprint_vector).encode()
        ).hexdigest()

        original_hash = get_blockchain_client().get_hash(video_id)
        if original_hash is None:
            return jsonify({
                "error": f"No blockchain record found for video_id '{video_id}'.",
                "tip":   "Register the original video first using /api/register",
            }), 404

        hash_match = new_hash == original_hash

        return jsonify({
            "success":  True,
            "video_id": video_id,
            "deepfake": _format_deepfake(deepfake_result),
            "hash_check": {
                "match":         hash_match,
                "original_hash": original_hash,
                "new_hash":      new_hash,
                "verdict":       "AUTHENTIC" if hash_match else "TAMPERED",
            },
            "overall": {
                "authentic": hash_match and deepfake_result["risk_level"] == "LOW",
                "verdict":   _combined_verdict(hash_match, deepfake_result["risk_level"]),
                "summary":   _combined_summary(hash_match, deepfake_result["risk_level"]),
            },
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


@app.route("/api/lookup/<video_id>", methods=["GET"])
def lookup_hash(video_id):
    try:
        stored = get_blockchain_client().get_hash(video_id)
        if stored is None:
            return jsonify({"error": "Not found"}), 404
        return jsonify({"video_id": video_id, "hash": stored})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_deepfake(result: dict) -> dict:
    """Format deepfake result for API response."""
    scores = result.get("frame_scores", [])
    top5   = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:5]
    return {
        "score":             result["score"],
        "verdict":           result["verdict"],
        "risk_level":        result["risk_level"],
        "confidence":        result["confidence"],
        "frames_analysed":   result["frames_analysed"],
        "is_trained":        result["is_trained"],
        "suspicious_frames": [{"frame": i, "score": round(s, 4)} for i, s in top5],
    }


def _combined_verdict(hash_match: bool, risk: str) -> str:
    if hash_match and risk == "LOW":
        return "FULLY AUTHENTIC"
    if hash_match and risk == "MEDIUM":
        return "HASH MATCH — BUT SUSPICIOUS"
    if hash_match and risk == "HIGH":
        return "HASH MATCH — DEEPFAKE DETECTED"
    if not hash_match and risk == "HIGH":
        return "TAMPERED AND DEEPFAKE DETECTED"
    if not hash_match:
        return "TAMPERED — HASH MISMATCH"
    return "INCONCLUSIVE"


def _combined_summary(hash_match: bool, risk: str) -> str:
    if hash_match and risk == "LOW":
        return "Both layers passed. This video is authentic."
    if hash_match and risk in ("MEDIUM", "HIGH"):
        return "Hash matches but AI found suspicious patterns. Manual review recommended."
    if not hash_match and risk == "HIGH":
        return "Critical: video tampered AND deepfake artifacts detected."
    if not hash_match:
        return "Video does not match its registered hash — it has been modified."
    return "Results inconclusive. Please review manually."


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
