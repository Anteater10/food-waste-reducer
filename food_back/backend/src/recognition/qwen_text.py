# backend/src/recognition/qwen_text.py

from typing import List, Dict, Optional
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_TEXT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_TEXT_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

_text_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    torch_dtype=_TEXT_DTYPE,
)
_text_model.to(_TEXT_DEVICE)


def _build_prompt(
    yolo_ingredients: List[Dict],
    blip_ingredients: List[Dict],
    vlm_ingredients_text: Optional[str],
) -> str:
    yolo_lines = [
        f"- name: {ing['name']}, confidence: {float(ing.get('confidence', 0.0)):.2f}"
        for ing in yolo_ingredients
    ]
    yolo_block = "\n".join(yolo_lines) if yolo_lines else "None"

    blip_lines = [
        f"- name: {ing['name']}, confidence: {float(ing.get('confidence', 0.0)):.2f}"
        for ing in blip_ingredients
    ]
    blip_block = "\n".join(blip_lines) if blip_lines else "None"

    vlm_block = vlm_ingredients_text or "None"

    prompt = f"""
You are an assistant that normalizes noisy vision outputs into a clean ingredient list.

You are given three sources of information for ONE kitchen photo:

1) YOLO-derived ingredients (object labels converted to ingredient names):
{yolo_block}

2) BLIP-derived ingredient-like phrases from the image caption:
{blip_block}

3) A vision-language model's raw guess of ingredients (comma-separated or free text):
{vlm_block}

TASK:
- Infer which FOOD INGREDIENTS are actually present.
- Normalize similar items to a canonical ingredient name, e.g.:
  - "granny smith apples" -> "apple"
  - "whole milk" -> "milk"
  - "cheddar cheese", "shredded cheese" -> "cheese" (if it seems like the same ingredient type)
- Ignore non-food items (fork, plate, bottle when contents are unknown).
- Do NOT invent ingredients that are not strongly suggested by any source.
- Exclude containers unless contents are clear (e.g. "jar of pickles" -> "pickles").

OUTPUT STRICTLY AS JSON with this schema:
{{
  "ingredients": [
    {{"name": "ingredient_name_1"}},
    {{"name": "ingredient_name_2"}},
    ...
  ]
}}

No explanations, no comments, no Markdown. Just valid JSON.
"""
    return prompt.strip()


def normalize_ingredients_to_json(
    yolo_ingredients: List[Dict],
    blip_ingredients: List[Dict],
    vlm_ingredients_text: Optional[str],
) -> Dict:
    """
    Returns {"ingredients": [{"name": "milk"}, ...]}.
    """
    prompt = _build_prompt(yolo_ingredients, blip_ingredients, vlm_ingredients_text)

    messages = [{"role": "user", "content": prompt}]
    chat_text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = _tokenizer([chat_text], return_tensors="pt")
    inputs = {k: v.to(_TEXT_DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _text_model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    gen_ids = outputs[0][len(inputs["input_ids"][0]):]
    raw = _tokenizer.decode(gen_ids, skip_special_tokens=True)
    raw_stripped = raw.strip()

    if raw_stripped.startswith("```"):
        lines = raw_stripped.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        raw_stripped = "\n".join(lines).strip()

    def _safe_parse(text: str) -> Dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except Exception:
                    return {"ingredients": []}
            return {"ingredients": []}

    data = _safe_parse(raw_stripped)

    if "ingredients" not in data or not isinstance(data["ingredients"], list):
        data["ingredients"] = []

    for ing in data["ingredients"]:
        if isinstance(ing, dict) and "name" in ing:
            ing["name"] = ing["name"].strip().lower()

    return data
