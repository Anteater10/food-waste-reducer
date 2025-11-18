from typing import List, Dict

# Numeric thresholds only (no semantic lists)
DEFAULT_MIN_CONFIDENCE = 0.20
DEFAULT_MIN_AREA_RATIO = 0.0025  # ~0.25% of image area


def detections_to_ingredients(
    detections: List[Dict],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_area_ratio: float = DEFAULT_MIN_AREA_RATIO,
) -> List[Dict]:
    """
    Convert YOLO detections into [{name, confidence}].

    Steps:
    - Filter by confidence.
    - Filter by bounding-box area ratio.
    - Use YOLO label as `name`.
    - Dedupe by name (keep highest confidence).
    """
    filtered = []

    for det in detections:
        label = det.get("label", "")
        conf = float(det.get("confidence", 0.0))
        area_ratio = float(det.get("area_ratio", 1.0))

        if not label:
            continue
        if conf < min_confidence:
            continue
        if area_ratio < min_area_ratio:
            continue

        filtered.append({"name": label, "confidence": conf})

    # Dedupe: keep highest-confidence per name
    deduped: Dict[str, Dict] = {}
    for ing in filtered:
        name = ing["name"]
        conf = ing["confidence"]
        if name not in deduped or conf > deduped[name]["confidence"]:
            deduped[name] = ing

    return list(deduped.values())
