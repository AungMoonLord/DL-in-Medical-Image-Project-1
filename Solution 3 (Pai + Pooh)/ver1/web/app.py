"""
Flask Web Application for Thai Character Classification Inference.

Features:
- REST API for live inference (Base64 drawing, File Upload, Test Samples).
- Model switching (CustomGlyphCNN, AdaptedResNet18, AdaptedMobileNetV3).
- Rich metadata rendering (Thai Name, English Meaning, TIS-620 hex code).
- Real-time letterbox tensor preview.
"""

import base64
import io
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image
import torch

# Ensure src is in python path
import sys
base_dir = Path(__file__).resolve().parent.parent
src_dir = base_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dataset import build_split_dataframes, tis620_to_char
from inference import THAI_CHAR_METADATA, ThaiCharacterInferenceEngine

app = Flask(
    __name__,
    template_folder=str(base_dir / "web" / "templates"),
    static_folder=str(base_dir / "web" / "static"),
)

DATASET_DIR = (base_dir / ".." / ".." / ".." / "ThaiCharacter Dataset" / "round2").resolve()
CHECKPOINTS_DIR = base_dir / "checkpoints"

# Initialize Inference Engine
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
engine = ThaiCharacterInferenceEngine(dataset_dir=DATASET_DIR, device=device)

# Load available model checkpoints
MODELS_CATALOG = [
    {
        "id": "custom_cnn",
        "name": "Custom GlyphCNN",
        "badge": "Custom Baseline",
        "description": "4-Stage Conv-BN-Mish-SE designed natively for 32x32 character glyphs.",
        "ckpt_path": CHECKPOINTS_DIR / "custom_cnn" / "best_model.pt",
    },
    {
        "id": "resnet18",
        "name": "Adapted ResNet-18",
        "badge": "Transfer Learning",
        "description": "ResNet-18 with stem adaptation and ImageNet pre-trained representations.",
        "ckpt_path": CHECKPOINTS_DIR / "resnet18" / "best_model.pt",
    },
    {
        "id": "mobilenet_v3",
        "name": "Adapted MobileNetV3",
        "badge": "Ultra Lightweight",
        "description": "Inverted residual bottlenecks with SE attention and high parameter efficiency.",
        "ckpt_path": CHECKPOINTS_DIR / "mobilenet_v3" / "best_model.pt",
    },
]

# Load models if checkpoint exists, otherwise initialize
for m_info in MODELS_CATALOG:
    m_id = m_info["id"]
    ckpt = m_info["ckpt_path"]
    if ckpt.exists():
        try:
            engine.load_model(m_id, ckpt)
            m_info["status"] = "Loaded (Trained)"
        except Exception as e:
            m_info["status"] = f"Error: {e}"
    else:
        # Initialize architecture without weights for demo if training in progress
        from models import build_model
        model = build_model(m_id, num_classes=engine.num_classes, pretrained=False)
        model = model.to(device).eval()
        engine.models[m_id] = model
        m_info["status"] = "Initialized"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/models", methods=["GET"])
def get_models():
    """Returns list of available models and active hardware device."""
    # Refresh catalog status
    catalog = []
    for m in MODELS_CATALOG:
        m_id = m["id"]
        is_trained = (m["ckpt_path"].exists())
        catalog.append({
            "id": m_id,
            "name": m["name"],
            "badge": m["badge"],
            "description": m["description"],
            "trained": is_trained,
            "status": "Ready (Trained Checkpoint)" if is_trained else "Initialized",
        })

    return jsonify({
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "models": catalog,
        "default_model": "custom_cnn",
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Inference endpoint.
    Accepts JSON body: { "image": "data:image/png;base64,...", "model": "custom_cnn", "top_k": 5 }
    Or multipart form data with 'file' and 'model'.
    """
    t0 = time.time()
    try:
        model_name = "custom_cnn"
        top_k = 5
        image_data = None

        if request.is_json:
            data = request.get_json()
            image_data = data.get("image")
            model_name = data.get("model", "custom_cnn")
            top_k = int(data.get("top_k", 5))
        elif "file" in request.files:
            file = request.files["file"]
            model_name = request.form.get("model", "custom_cnn")
            top_k = int(request.form.get("top_k", 5))
            image_data = file.read()
        elif "image" in request.form:
            image_data = request.form["image"]
            model_name = request.form.get("model", "custom_cnn")

        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        # Reload trained checkpoint if it became available on disk
        ckpt_path = CHECKPOINTS_DIR / model_name / "best_model.pt"
        if ckpt_path.exists() and (model_name not in engine.models or engine.models[model_name] is None):
            engine.load_model(model_name, ckpt_path)

        result = engine.predict(image_data, model_name=model_name, top_k=top_k)
        result["latency_ms"] = round((time.time() - t0) * 1000, 2)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sample_characters", methods=["GET"])
def get_sample_characters():
    """Returns a list of random character samples from the dataset for 1-click testing."""
    samples = []
    class_dirs = [d for d in DATASET_DIR.iterdir() if d.is_dir() and d.name.isdigit()]
    selected_dirs = random.sample(class_dirs, min(12, len(class_dirs)))

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
    print(f"🌐 Starting Thai Character Classification Inference Web App on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
