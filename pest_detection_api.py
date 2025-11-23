#!/usr/bin/env python3
"""
Lightweight Flask API for multi-pest detection and counting

Endpoints:
- GET /health → status check
- POST /detect (multipart form-data: image=<file>) → per-class counts

Loads YOLO model once at startup for fast inference.
"""

from __future__ import annotations

import io
import os
import time
import requests
from typing import Dict, Any
from pathlib import Path

from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO


app = Flask(__name__)


# === Configuration ===
# Get active model path from database via PHP endpoint
def get_active_model_path() -> str:
    """Fetch the currently active model path from the database"""
    try:
        # Call PHP endpoint to get active model
        # Updated path for AgriShield_ML_Flask location
        response = requests.get('http://localhost/Proto1/pest_detection_ml/api/get_active_model_path.php', timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                model_path = data.get('model_path')
                if model_path and os.path.exists(model_path):
                    print(f"✅ Using active model: {os.path.basename(model_path)}")
                    return model_path
    except Exception as e:
        print(f"⚠️ Could not get active model from database: {e}")
    
    # Fallback to default model - using "best 2.pt" from datasets folder
    # Updated paths for AgriShield_ML_Flask location (one level up from Proto1)
    base_dir = Path(__file__).resolve().parent.parent
    datasets_model = base_dir / "datasets" / "best 2.pt"
    if datasets_model.exists():
        print(f"✅ Using default model: best 2.pt (from datasets folder)")
        return str(datasets_model)
    
    # Secondary fallback to models folder
    fallback = base_dir / "pest_detection_ml" / "models" / "best.pt"
    if fallback.exists():
        print(f"⚠️ Using fallback model: best.pt (from models folder)")
        return str(fallback)
    
    # If neither exists, return datasets path (will show error when loading)
    print(f"❌ Model not found: {datasets_model}")
    return str(datasets_model)

# Get model path (will use active model from database)
MODEL_PATH = get_active_model_path()

# Class names will be loaded dynamically from the model
CLASS_NAMES = []


def load_model() -> YOLO:
    global CLASS_NAMES
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_PATH}. Ensure training completed and path is correct."
        )
    model = YOLO(MODEL_PATH)
    
    # Get class names dynamically from the model
    if hasattr(model, 'names') and model.names:
        CLASS_NAMES = [model.names[i] for i in sorted(model.names.keys())]
        print(f"✅ Loaded {len(CLASS_NAMES)} classes from model: {CLASS_NAMES}")
    else:
        # Fallback to default if model doesn't have names
        CLASS_NAMES = [
            "Rice_Bug",
            "black-bug",
            "brown_hopper",
            "green_hopper",
        ]
        print(f"⚠️ Model doesn't have class names, using fallback: {CLASS_NAMES}")
    
    return model


# Model will be loaded lazily on first request
model = None


@app.get("/health")
def health() -> Any:
    global model, CLASS_NAMES
    # Load model if not loaded to get class names
    if model is None:
        model = load_model()
    return jsonify({
        "status": "ok",
        "model": os.path.basename(MODEL_PATH),
        "classes": CLASS_NAMES if CLASS_NAMES else ["Loading..."],
        "num_classes": len(CLASS_NAMES) if CLASS_NAMES else 0,
    })


def aggregate_counts(pred) -> Dict[str, int]:
    """Aggregate per-class counts from a single prediction result."""
    counts: Dict[str, int] = {name: 0 for name in CLASS_NAMES}
    if pred is None or pred.boxes is None:
        return counts
    cls_tensor = pred.boxes.cls
    if cls_tensor is None:
        return counts
    for c in cls_tensor.tolist():
        idx = int(c)
        if 0 <= idx < len(CLASS_NAMES):
            counts[CLASS_NAMES[idx]] += 1
    return counts


@app.post("/detect")
def detect() -> Any:
    global model, CLASS_NAMES
    if model is None:
        model = load_model()
        # Ensure CLASS_NAMES is populated after model load
        if not CLASS_NAMES:
            # Fallback if model doesn't have names
            CLASS_NAMES = [
                "Rice_Bug",
                "black-bug",
                "brown_hopper",
                "green_hopper",
            ]
    
    if "image" not in request.files:
        return jsonify({"error": "missing file field 'image'"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    try:
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"invalid image: {e}"}), 400

    t0 = time.time()
    # Run inference; use slightly higher NMS to suppress duplicates
    results = model.predict(img, imgsz=512, conf=0.15, iou=0.50, device="cpu")
    dt = time.time() - t0

    # Dynamic confidence thresholds - defaults for common pests
    # Can be customized per model
    default_threshold = 0.25
    class_conf_thresholds = {
        "Rice_Bug": 0.20,
        "black-bug": 0.80,       # Very high threshold to reduce black-bug bias
        "brown_hopper": 0.15,
        "green_hopper": 0.15,
        "stem_borer": 0.20,
        "white_stem_borer": 0.20,
        "White_Stem_Borer": 0.20,
    }
    # Use default threshold for any class not in the dictionary

    # Optional: allow disabling black-bug via query param ?disable_black=1
    if request.args.get("disable_black") == "1":
        class_conf_thresholds["black-bug"] = 1.0

    # Aggregate counts with class-specific filtering
    counts: Dict[str, int] = {name: 0 for name in CLASS_NAMES}
    pred = results[0] if results else None
    if pred is not None and getattr(pred, "boxes", None) is not None:
        boxes = pred.boxes
        try:
            num = len(boxes)
        except Exception:
            num = 0
        for i in range(num):
            try:
                cls_idx = int(boxes.cls[i].item()) if hasattr(boxes.cls, 'shape') else int(boxes.cls.tolist()[i])
                conf = float(boxes.conf[i].item()) if hasattr(boxes.conf, 'shape') else float(boxes.conf.tolist()[i])
                if 0 <= cls_idx < len(CLASS_NAMES):
                    name = CLASS_NAMES[cls_idx]
                    # Apply class-specific confidence filtering
                    threshold = class_conf_thresholds.get(name, 0.25)
                    if conf >= threshold:
                        counts[name] += 1
            except Exception:
                continue

    # Dynamic pesticide recommendations - can be extended for any pest class
    pesticide_recs = {
        "Rice_Bug": "Use lambda-cyhalothrin or beta-cyfluthrin per label; avoid spraying near harvest.",
        "green_hopper": "Imidacloprid or dinotefuran early; rotate MoA to avoid resistance.",
        "brown_hopper": "Buprofezin or pymetrozine; reduce nitrogen; avoid broad-spectrum pyrethroids.",
        "black-bug": "Carbaryl dust or fipronil bait at tillering; field sanitation recommended.",
        "stem_borer": "Chlorantraniliprole or fipronil at early stage; remove stubble after harvest.",
        "white_stem_borer": "Chlorantraniliprole or fipronil at early stage; remove stubble after harvest.",
        "White_Stem_Borer": "Chlorantraniliprole or fipronil at early stage; remove stubble after harvest.",
    }

    # Dynamic pest diseases - can be extended for any pest class
    pest_diseases = {
        "Rice_Bug": [
            "Grain discoloration and shriveling (sucking damage)",
            "Reduced grain filling leading to yield loss"
        ],
        "green_hopper": [
            "Transmits rice tungro virus (RTV)",
            "Hopperburn under high populations"
        ],
        "brown_hopper": [
            "Hopperburn (sudden plant wilting)",
            "Can facilitate sooty mold via honeydew"
        ],
        "black-bug": [
            "Tillering stage damage (sucking), resulting in stunted growth",
            "White earheads in severe infestations"
        ],
        "stem_borer": [
            "Dead hearts (at tillering stage)",
            "White heads (at heading stage)"
        ],
        "white_stem_borer": [
            "Dead hearts (at tillering stage)",
            "White heads (at heading stage)"
        ],
        "White_Stem_Borer": [
            "Dead hearts (at tillering stage)",
            "White heads (at heading stage)"
        ],
    }

    # Filter recommendations to only detected pests (>0)
    recommendations = {k: v for k, v in pesticide_recs.items() if counts.get(k, 0) > 0}
    diseases = {k: pest_diseases.get(k, []) for k, v in counts.items() if v > 0}

    response_payload = {
        "status": "success",
        "pest_counts": counts,
        "diseases": diseases,
        "recommendations": recommendations,
        "inference_time_ms": round(dt * 1000, 1),
        "model": os.path.basename(MODEL_PATH),
    }

    # Optional debug: return raw detections with classes and confidences
    if request.args.get("debug") == "1" and results and results[0] and getattr(results[0], "boxes", None):
        dets = []
        try:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_idx = int(boxes.cls[i].item()) if hasattr(boxes.cls, 'shape') else int(boxes.cls.tolist()[i])
                conf = float(boxes.conf[i].item()) if hasattr(boxes.conf, 'shape') else float(boxes.conf.tolist()[i])
                name = CLASS_NAMES[cls_idx] if 0 <= cls_idx < len(CLASS_NAMES) else str(cls_idx)
                dets.append({"class": name, "conf": round(conf, 3)})
        except Exception:
            pass
        response_payload["detections"] = dets

    return jsonify(response_payload)


if __name__ == "__main__":
    print("Starting Flask API...")
    # Bind to all interfaces so mobile or LAN clients can reach it
    port = int(os.environ.get("PORT", "5001"))
    print(f"Running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)



