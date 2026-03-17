"""
AI-Driven Deepfake Insurance & Forensic Ledger
================================================
Flask backend — handles video upload, AI fingerprinting,
SHA-256 hashing, blockchain interaction, and verification.
"""

import os
import json
import hashlib
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB limit
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ---------------------------------------------------------------------------
# Lazy imports — heavy ML/blockchain libs load only when first used
# ---------------------------------------------------------------------------
_video_processor = None
_ai_fingerprinter = None
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


def get_blockchain_client():
    global _blockchain_client
    if _blockchain_client is None:
        from contracts.blockchain_client import BlockchainClient
        _blockchain_client = BlockchainClient()
    return _blockchain_client


def allowed_file(filename: str) -> bool:
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
    POST /api/register
    ------------------
    Upload a video, generate its AI fingerprint + SHA-256 hash,
    and store the hash on the blockchain.

    Returns JSON with:
      - video_id      : unique identifier for this registration
      - fingerprint   : serialised AI feature vector (truncated preview)
      - hash          : SHA-256 hex digest
      - tx_hash       : blockchain transaction hash
      - block_number  : block where hash was recorded
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid or unsupported file type"}), 400

    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(video_path)

    try:
        # Step 1 — Extract frames & metadata  (Member 1)
        processor = get_video_processor()
        frames, metadata = processor.extract(video_path)

        # Step 2 — Generate AI fingerprint     (Member 2)
        fingerprinter = get_ai_fingerprinter()
        fingerprint_vector = fingerprinter.fingerprint(frames)

        # Step 3 — SHA-256 hash the fingerprint
        fingerprint_bytes = json.dumps(fingerprint_vector).encode("utf-8")
        video_hash = hashlib.sha256(fingerprint_bytes).hexdigest()

        # Step 4 — Store hash on blockchain    (Member 3)
        blockchain = get_blockchain_client()
        video_id = hashlib.md5(filename.encode()).hexdigest()[:12]
        tx_receipt = blockchain.store_hash(video_id, video_hash)

        return jsonify({
            "success": True,
            "video_id": video_id,
            "filename": filename,
            "metadata": metadata,
            "fingerprint_preview": fingerprint_vector[:8],   # first 8 dims only
            "hash": video_hash,
            "tx_hash": tx_receipt["tx_hash"],
            "block_number": tx_receipt["block_number"],
            "explorer_url": tx_receipt.get("explorer_url", ""),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up uploaded file after processing
        if os.path.exists(video_path):
            os.remove(video_path)


@app.route("/api/verify", methods=["POST"])
def verify_video():
    """
    POST /api/verify
    ----------------
    Re-upload a video and a known video_id to verify authenticity.
    Generates a fresh fingerprint + hash, then compares against
    the hash stored on the blockchain.

    Returns JSON with:
      - authentic     : True / False
      - original_hash : hash stored on blockchain
      - new_hash      : hash computed from uploaded video
      - match         : True if hashes are identical
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_id = request.form.get("video_id", "").strip()
    if not video_id:
        return jsonify({"error": "video_id is required"}), 400

    file = request.files["video"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], f"verify_{filename}")
    file.save(video_path)

    try:
        # Re-run the same pipeline on the candidate video
        processor = get_video_processor()
        frames, _ = processor.extract(video_path)

        fingerprinter = get_ai_fingerprinter()
        fingerprint_vector = fingerprinter.fingerprint(frames)

        fingerprint_bytes = json.dumps(fingerprint_vector).encode("utf-8")
        new_hash = hashlib.sha256(fingerprint_bytes).hexdigest()

        # Fetch original hash from blockchain
        blockchain = get_blockchain_client()
        original_hash = blockchain.get_hash(video_id)

        if original_hash is None:
            return jsonify({"error": f"No record found for video_id '{video_id}'"}), 404

        match = new_hash == original_hash

        return jsonify({
            "success": True,
            "video_id": video_id,
            "authentic": match,
            "match": match,
            "original_hash": original_hash,
            "new_hash": new_hash,
            "verdict": "AUTHENTIC — hashes match" if match else "TAMPERED — hashes differ",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


@app.route("/api/lookup/<video_id>", methods=["GET"])
def lookup_hash(video_id):
    """GET /api/lookup/<video_id> — retrieve stored hash for a given video ID."""
    try:
        blockchain = get_blockchain_client()
        stored_hash = blockchain.get_hash(video_id)
        if stored_hash is None:
            return jsonify({"error": "Not found"}), 404
        return jsonify({"video_id": video_id, "hash": stored_hash})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
