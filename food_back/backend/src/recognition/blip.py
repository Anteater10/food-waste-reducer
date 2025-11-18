from typing import List, Dict
from PIL import Image
import io
import re

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP once at import time
_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
_blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to("cuda" if torch.cuda.is_available() else "cpu")


def caption_image(image_bytes: bytes) -> str:
    """
    Run BLIP image captioning on the full image and return a single caption string.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = _processor(image, return_tensors="pt")
    inputs = {k: v.to(_blip_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = _blip_model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=3,
        )

    caption = _processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()


def caption_to_ingredients(caption: str) -> List[Dict]:
    """
    Convert a BLIP caption into a rough list of ingredient phrases.
    """
    text = caption.lower()

    # Split on commas and " and "
    parts = re.split(r",| and ", text)

    ingredients = []
    for part in parts:
        phrase = part.strip()
        if not phrase:
            continue

        ingredients.append(
            {
                "name": phrase,
                # heuristic confidence
                "confidence": 0.75,
            }
        )

    # Dedupe by name
    deduped: Dict[str, Dict] = {}
    for ing in ingredients:
        name = ing["name"]
        conf = ing["confidence"]
        if name not in deduped or conf > deduped[name]["confidence"]:
            deduped[name] = ing

    return list(deduped.values())
