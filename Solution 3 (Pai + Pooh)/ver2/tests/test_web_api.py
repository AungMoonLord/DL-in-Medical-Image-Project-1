import sys
import io
import base64
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from PIL import Image

sol3_root = Path(__file__).resolve().parent.parent.parent
src_dir = Path(__file__).resolve().parent.parent / "src"
web_dir = sol3_root / "web"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(web_dir) not in sys.path:
    sys.path.insert(0, str(web_dir))

from app import app, engine


def test_api():
    client = app.test_client()

    # 1. Test GET /api/models
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.get_json()
    print("✓ Models API returned:", len(data["models"]), "models.")

    # 2. Test GET /api/sample_characters
    res = client.get("/api/sample_characters")
    assert res.status_code == 200
    data = res.get_json()
    print("✓ Sample characters API returned:", len(data["samples"]), "samples.")

    # 3. Test POST /api/predict with dummy character drawing
    test_img = Image.new("RGB", (280, 280), (255, 255, 255))
    buf = io.BytesIO()
    test_img.save(buf, format="PNG")
    b64_img = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    res = client.post("/api/predict", json={
        "image": b64_img,
        "model": "custom_cnn_v1" if "custom_cnn_v1" in engine.models else list(engine.models.keys())[0],
        "top_k": 5,
        "pad_ratio": 0.15,
        "use_focal_cleaner": True,
    })
    assert res.status_code == 200
    pred_data = res.get_json()
    assert "top_prediction" in pred_data
    assert "preprocessed_preview" in pred_data
    print("✓ Predict API succeeded. Top prediction:", pred_data["top_prediction"]["name_th"], "in", pred_data["latency_ms"], "ms.")

    # 4. Test POST /api/load_custom_checkpoint with Solution 2 checkpoint
    sol2_ckpt = Path(__file__).resolve().parent.parent.parent / "Solution 2 (Eungul + Ninenine)" / "best_thai_character_model.pth"
    if sol2_ckpt.exists():
        res = client.post("/api/load_custom_checkpoint", json={
            "checkpoint_path": str(sol2_ckpt),
            "model_name": "TestSol2EffNet",
        })
        assert res.status_code == 200
        load_data = res.get_json()
        assert load_data["success"] is True
        print("✓ Universal custom checkpoint loader successfully loaded Solution 2 EfficientNet!")

    print("\n🎉 ALL WEB API ENDPOINTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_api()
