from ultralytics import YOLO
from PIL import Image
import io

# Load YOLO model once at startup
_model = YOLO("yolov8l.pt")


def detect(image_bytes: bytes):
    """
    Run YOLO on the image and return a list of:
    {
      "label": <str>,
      "confidence": <float>,
      "area_ratio": <float>
    }
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    image_area = float(width * height) if width > 0 and height > 0 else 1.0

    results = _model(image, verbose=False)

    detections = []
    first = results[0]
    boxes = first.boxes
    names = first.names  # {class_id: class_name}

    if boxes is None:
        return detections

    xyxy_list = boxes.xyxy.tolist()
    class_ids = boxes.cls.tolist()
    confidences = boxes.conf.tolist()

    for (x1, y1, x2, y2), cls_id, conf in zip(xyxy_list, class_ids, confidences):
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        area_ratio = (w * h) / image_area

        detections.append(
            {
                "label": names[int(cls_id)],
                "confidence": float(conf),
                "area_ratio": float(area_ratio),
            }
        )

    return detections
