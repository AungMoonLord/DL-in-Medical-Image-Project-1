"""
Flask Web Application for Thai Character Classification Inference (ver2).

Features:
- REST API for live inference (Canvas Drawing, File Upload, Dataset Samples).
- Universal Model Switching: Auto-detects and loads CustomGlyphCNN, ResNet-18, MobileNetV3, Solution 2 EfficientNet-B0.
- Custom Checkpoint Selector: Direct path loading from filesystem.
- Drawing Pad Config: Canvas resolution selector, Pixel/Binary brush mode, Adaptive bounding-box padding.
- Real-time Multi-Stage Tensor Preview (Raw Drawing -> Bounding Box Crop -> Letterbox Padded Tensor).
- Rich Metadata rendering (Thai Name, English Meaning, TIS-620 Hex, Linguistic Type).
"""

import base64
import io
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request
from PIL import Image
import torch

# Setup path directories
web_dir = Path(__file__).resolve().parent
sol3_root = web_dir.parent
ver2_dir = sol3_root / "ver2"
src_dir = ver2_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dataset import build_split_dataframes, tis620_to_char
from inference import THAI_CHAR_METADATA, ThaiCharacterInferenceEngine

app = Flask(
    __name__,
    template_folder=str(web_dir / "templates"),
    static_folder=str(web_dir / "static"),
)

# Search candidate dataset directories
dataset_candidates = [
    (sol3_root / ".." / ".." / "ThaiCharacterDataset" / "round2-cleaned").resolve(),
    (sol3_root / ".." / ".." / "ThaiCharacter Dataset" / "round2-cleaned").resolve(),
    (sol3_root / ".." / ".." / "ThaiCharacterDataset" / "round2").resolve(),
    (sol3_root / ".." / ".." / "ThaiCharacter Dataset" / "round2").resolve(),
]
DATASET_DIR = next((p for p in dataset_candidates if p.exists()), dataset_candidates[0])
CHECKPOINTS_DIR = ver2_dir / "checkpoints"
SOL1_DIR = (sol3_root / ".." / "Solution 1 (AUNG + BAM)").resolve()
SOL2_DIR = (sol3_root / ".." / "Solution 2 (Eungul + Ninenine)").resolve()
V1_DIR = (sol3_root / "ver1").resolve()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
engine = ThaiCharacterInferenceEngine(dataset_dir=DATASET_DIR, device=device)

# Catalog of known model checkpoints
MODELS_CATALOG = [
    {
        "id": "custom_cnn_v2",
        "name": "Custom GlyphCNN (ver2)",
        "badge": "Native Baseline (32px)",
        "description": "4-Stage Conv-BN-Mish-SE designed natively for 32x32 character glyphs.",
        "ckpt_path": CHECKPOINTS_DIR / "custom_cnn" / "best_model.pt",
        "default": True,
    },
    {
        "id": "custom_cnn_v1",
        "name": "Custom GlyphCNN (ver1 Legacy)",
        "badge": "v1 Baseline",
        "description": "v1 Custom CNN trained on uncleaned round2 dataset.",
        "ckpt_path": V1_DIR / "checkpoints" / "custom_cnn" / "best_model.pt",
        "default": False,
    },
    {
        "id": "solution2_effnet",
        "name": "Solution 2 (EfficientNet-B0)",
        "badge": "Solution 2 (224px)",
        "description": "EfficientNet-B0 transfer learning checkpoint contributed by Eungul + Ninenine.",
        "ckpt_path": SOL2_DIR / "best_thai_character_model.pth",
        "default": False,
    },
    {
        "id": "resnet18",
        "name": "Adapted ResNet-18",
        "badge": "Transfer Learning (64px)",
        "description": "ResNet-18 with 3x3 stride-1 stem adaptation and ImageNet pre-training.",
        "ckpt_path": CHECKPOINTS_DIR / "resnet18" / "best_model.pt",
        "default": False,
    },
    {
        "id": "mobilenet_v3",
        "name": "Adapted MobileNetV3",
        "badge": "Ultra Lightweight (64px)",
        "description": "Inverted residual bottlenecks with SE attention and high parameter efficiency.",
        "ckpt_path": CHECKPOINTS_DIR / "mobilenet_v3" / "best_model.pt",
        "default": False,
    },
]

# Auto-load available checkpoints at startup
for m in MODELS_CATALOG:
    m_id = m["id"]
    ckpt = Path(m["ckpt_path"])
    if ckpt.exists():
        try:
            engine.load_model(m_id, ckpt)
            m["status"] = "Ready (Trained)"
        except Exception as e:
            m["status"] = f"Error: {e}"
    else:
        m["status"] = "Checkpoint not found on disk"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/models", methods=["GET"])
def get_models():
    """Returns list of all available models and active hardware device."""
    catalog = []
    for m in MODELS_CATALOG:
        m_id = m["id"]
        is_loaded = m_id in engine.models
        is_on_disk = Path(m["ckpt_path"]).exists()
        cfg = engine.model_configs.get(m_id, {})
        catalog.append({
            "id": m_id,
            "name": m["name"],
            "badge": m["badge"],
            "description": m["description"],
            "ckpt_path": str(m["ckpt_path"]),
            "trained": is_on_disk,
            "loaded": is_loaded,
            "resolution": f"{cfg.get('target_res', 32)}x{cfg.get('target_res', 32)}",
            "model_type": cfg.get("type", "unknown"),
            "status": "Active & Ready" if is_loaded else ("Ready to Load" if is_on_disk else "Not Found"),
        })

    return jsonify({
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "models": catalog,
        "default_model": "custom_cnn_v1" if "custom_cnn_v1" in engine.models else "custom_cnn_v2",
    })


@app.route("/api/load_custom_checkpoint", methods=["POST"])
def load_custom_checkpoint():
    """Dynamically loads any checkpoint file from a user-specified file path."""
    try:
        data = request.get_json() or {}
        ckpt_path_str = data.get("checkpoint_path", "").strip()
        custom_name = data.get("model_name", "").strip() or Path(ckpt_path_str).stem

        if not ckpt_path_str:
            return jsonify({"error": "No checkpoint path provided"}), 400

        ckpt_path = Path(ckpt_path_str).resolve()
        if not ckpt_path.exists():
            return jsonify({"error": f"File does not exist at: {ckpt_path}"}), 404

        engine.load_model(custom_name, ckpt_path)
        cfg = engine.model_configs.get(custom_name, {})

        # Add to catalog if not present
        if not any(m["id"] == custom_name for m in MODELS_CATALOG):
            MODELS_CATALOG.append({
                "id": custom_name,
                "name": f"Custom: {custom_name}",
                "badge": f"{cfg.get('type', 'Custom')} ({cfg.get('target_res', 32)}px)",
                "description": f"Loaded from {ckpt_path}",
                "ckpt_path": ckpt_path,
                "status": "Ready (Trained)",
            })

        return jsonify({
            "success": True,
            "model_id": custom_name,
            "model_type": cfg.get("type"),
            "target_resolution": cfg.get("target_res"),
            "message": f"Successfully loaded checkpoint ({cfg.get('type')}, {cfg.get('target_res')}x{cfg.get('target_res')})",
        })
    except Exception as e:
        return jsonify({"error": f"Failed to load checkpoint: {str(e)}"}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Inference endpoint.
    Accepts: { image: base64, model: str, top_k: int, pad_ratio: float, use_focal_cleaner: bool }
    """
    t0 = time.time()
    try:
        model_name = "custom_cnn_v2"
        top_k = 5
        pad_ratio = 0.15
        use_focal_cleaner = False
        image_data = None

        if request.is_json:
            data = request.get_json()
            image_data = data.get("image")
            model_name = data.get("model", "custom_cnn_v2")
            top_k = int(data.get("top_k", 5))
            pad_ratio = float(data.get("pad_ratio", 0.15))
            use_focal_cleaner = bool(data.get("use_focal_cleaner", False))
        elif "file" in request.files:
            file = request.files["file"]
            model_name = request.form.get("model", "custom_cnn_v2")
            top_k = int(request.form.get("top_k", 5))
            pad_ratio = float(request.form.get("pad_ratio", 0.15))
            use_focal_cleaner = bool(request.form.get("use_focal_cleaner", False))
            image_data = file.read()

        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        # Fallback if selected model is not loaded yet but checkpoint exists
        if model_name not in engine.models:
            for m in MODELS_CATALOG:
                if m["id"] == model_name and Path(m["ckpt_path"]).exists():
                    engine.load_model(model_name, Path(m["ckpt_path"]))
                    break

        if model_name not in engine.models:
            # Fallback to any loaded model
            if engine.models:
                model_name = list(engine.models.keys())[0]
            else:
                return jsonify({"error": "No models are currently loaded."}), 500

        result = engine.predict(
            image_data,
            model_name=model_name,
            top_k=top_k,
            pad_ratio=pad_ratio,
            use_focal_cleaner=use_focal_cleaner,
        )
        result["latency_ms"] = round((time.time() - t0) * 1000, 2)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sample_characters", methods=["GET"])
def get_sample_characters():
    """Returns random character samples from the cleaned dataset for 1-click testing."""
    samples = []
    class_dirs = [d for d in DATASET_DIR.iterdir() if d.is_dir() and d.name.isdigit()]
    selected_dirs = random.sample(class_dirs, min(14, len(class_dirs)))

    for c_dir in selected_dirs:
        class_num = int(c_dir.name)
        images = [f for f in c_dir.iterdir() if f.is_file() and f.suffix.lower() in (".jpg", ".png")]
        if images:
            img_file = random.choice(images)
            with Image.open(img_file) as im:
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

            meta = THAI_CHAR_METADATA.get(class_num, {
                "char": tis620_to_char(class_num),
                "name_th": f"รหัส {class_num}",
                "name_en": f"Class {class_num}",
            })

            samples.append({
                "class_number": class_num,
                "character": meta["char"],
                "name_th": meta["name_th"],
                "name_en": meta["name_en"],
                "image_b64": b64,
            })

    return jsonify({"samples": samples})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🌐 Starting Thai Character Classifier (ver2) Web App on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
