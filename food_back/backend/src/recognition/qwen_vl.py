# backend/src/recognition/qwen_vl.py

from typing import Optional
from PIL import Image
import io

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# Choose device + dtype
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Load Qwen3-VL once (no device_map="auto")
_qwen_vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    torch_dtype=_DTYPE,
)
_qwen_vl_model.to(_DEVICE)

_qwen_vl_processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")


def _bytes_to_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def describe_ingredients_with_vlm(
    image_bytes: bytes,
    extra_context: Optional[str] = None,
) -> str:
    """
    Use Qwen3-VL-4B-Instruct to list ingredients / food items in the image.

    Returns a raw string (e.g., "milk, eggs, spinach, yogurt").
    """
    image = _bytes_to_image(image_bytes)

    context = extra_context or ""
    prompt = (
        "You are helping build a structured ingredient inventory from kitchen photos. "
        "List all food items and ingredients you can see in this image as a comma-separated list. "
        "Use short, generic ingredient names (e.g., 'milk', 'eggs', 'spinach', 'yogurt'). "
        "Avoid describing colors, brands, or packaging unless needed to distinguish items. "
        f"{context}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = _qwen_vl_processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    # move tensors to same device as model
    inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = _qwen_vl_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    # Strip off prompt tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    text = _qwen_vl_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return text.strip()
