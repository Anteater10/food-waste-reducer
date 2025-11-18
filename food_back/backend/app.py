# backend/app.py

from flask import Flask, jsonify, request
from flask_cors import CORS

from src.recognition.model import detect as detect_objects
from src.recognition.ingredients import detections_to_ingredients
from src.recognition.blip import caption_image, caption_to_ingredients
from src.recognition.qwen_vl import describe_ingredients_with_vlm
from src.recognition.qwen_text import normalize_ingredients_to_json

app = Flask(__name__)
CORS(app)


@app.route("/api/hello")
def hello():
    return jsonify({"message": "Hello from backend"})


@app.route("/api/ingredients/detect", methods=["POST"])
def detect_ingredients():
    """
    Full ingredient detection pipeline (v2):

    1. Accept an uploaded image (field name: 'image').
    2. YOLOv8l -> object detections -> YOLO ingredients.
    3. BLIP -> full-image caption -> BLIP pseudo-ingredients.
    4. Qwen3-VL -> raw ingredient string.
    5. Qwen3-4B-Instruct -> normalize everything into canonical ingredients JSON.
    6. Return:
       - raw merged ingredients (YOLO+BLIP)
       - caption
       - vlm_raw
       - normalized canonical ingredients
    """
    if "image" not in request.files:
        return jsonify({"error": "Image not provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    image_bytes = file.read()

    # 1. YOLO-based detections
    yolo_detections = detect_objects(image_bytes)
    yolo_ingredients = detections_to_ingredients(yolo_detections)

    # 2. BLIP caption -> ingredient-like phrases
    caption = caption_image(image_bytes)
    blip_ingredients = caption_to_ingredients(caption)

    # 3. Vision LLM (Qwen3-VL) -> raw ingredient text
    vlm_raw = describe_ingredients_with_vlm(
        image_bytes,
        extra_context="If you are unsure, leave the item out rather than guessing.",
    )

    # 4. Merge YOLO + BLIP for a basic, model-agnostic list (your original behavior)
    merged_raw = {}
    # First YOLO
    for ing in yolo_ingredients:
        name = ing["name"]
        merged_raw[name] = {
            "name": name,
            "confidence": float(ing.get("confidence", 0.0)),
            "source": "yolo",
        }

    # Then BLIP
    for ing in blip_ingredients:
        name = ing["name"]
        conf = float(ing.get("confidence", 0.0))
        existing = merged_raw.get(name)

        if existing is None or conf > existing["confidence"]:
            merged_raw[name] = {
                "name": name,
                "confidence": conf,
                "source": "blip",
            }

    raw_ingredients_list = [
        {"name": v["name"], "confidence": v["confidence"]}
        for v in merged_raw.values()
    ]
    raw_ingredients_list.sort(key=lambda x: x["confidence"], reverse=True)

    # 5. Normalization via text LLM (Qwen3-4B-Instruct-2507)
    try:
        normalized = normalize_ingredients_to_json(
            yolo_ingredients=yolo_ingredients,
            blip_ingredients=blip_ingredients,
            vlm_ingredients_text=vlm_raw,
        )
    except Exception as e:
        # Failsafe: fall back to the raw merged ingredients
        normalized = {
            "ingredients": [
                {"name": ing["name"].lower()}
                for ing in raw_ingredients_list
            ]
        }

    return jsonify(
        {
            "ingredients_raw": raw_ingredients_list,     # YOLO+BLIP merged
            "caption": caption,                          # BLIP caption
            "vlm_raw": vlm_raw,                          # Qwen3-VL raw ingredient text
            "ingredients_normalized": normalized,        # canonical list via Qwen3-4B
        }
    )


if __name__ == "__main__":
    # For local dev; in production you'd use gunicorn/uvicorn/etc.
    app.run(host="0.0.0.0", port=5001, debug=True)
