#!/usr/bin/env python3
"""Standalone inference for Pakistani Car Damage AI — submission version.

WHAT THIS SCRIPT DOES (and does not do):
- Car detection: ALWAYS via a COCO pretrained YOLOv8n model (default: yolov8n.pt
  in the project root). This is a generic car detector, NOT the project model.
  Its results are labelled as such in the output JSON.
- Part segmentation: ONLY if a real fine-tuned model exists at
  models/part_detection/best.pt (the Colab notebook output). If that file is
  missing, part_detections is [] and part_model_status is "NOT_TRAINED" —
  no parts are invented, guessed, or hardcoded.
- Make/model/year and damage types: NOT predicted by either model above.
  They are reported as "Unknown" / null until a classifier/damage model exists.
  Do not edit this script to guess them.

CONNECTION MECHANISM (how Colab training connects to inference):
  1. Train in Colab (colab/Train_CarParts_Seg_YOLOv8n_Seg.ipynb).
  2. Copy the exported best.pt to models/part_detection/best.pt (exact path).
  3. Run this script again — it auto-detects best.pt, loads it as a segmentation
     model, and part_model_status becomes "TRAINED". No code change is needed.
  4. Use --weights to point at a different part-model file for testing.

Usage (from the project root):
  python predict.py --image uploads/images/honda_civic_cc0.jpg --output-dir outputs/predict_test
  python predict.py --image mycar.jpg --weights /path/to/best.pt --json-out result.json

Output: a JSON file (or stdout) + an annotated JPG in --output-dir.
  JSON fields: model_info, car_detections, part_detections, summary.
  Every confidence in the JSON comes from a model output. Nothing is simulated.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PART_WEIGHTS_DEFAULT = PROJECT_ROOT / "models" / "part_detection" / "best.pt"
CAR_WEIGHTS_DEFAULT = PROJECT_ROOT / "yolov8n.pt"
CAR_WEIGHTS_FALLBACK = PROJECT_ROOT / "models" / "car_detection" / "yolov8n.pt"


def find_car_weights(explicit):
    if explicit and Path(explicit).exists():
        return Path(explicit)
    for cand in (CAR_WEIGHTS_DEFAULT, CAR_WEIGHTS_FALLBACK):
        if cand.exists():
            return cand
    return None


def run_inference(image_path, part_weights=None, car_weights=None, conf=0.25, imgsz=640):
    """Run car detection + (if present) part segmentation. Returns (result_dict, annotated_image_array_or_None).

    Raises FileNotFoundError if the image or the car weights are missing.
    Never invents detections: empty model output means an empty list.
    """
    from ultralytics import YOLO  # imported here so --help works without torch installed

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    car_path = find_car_weights(car_weights)
    if car_path is None:
        raise FileNotFoundError(
            "Car detector weights not found. Expected yolov8n.pt in the project root "
            "(generic COCO detector) — see SUBMISSION_README.md."
        )
    part_path = Path(part_weights) if part_weights else PART_WEIGHTS_DEFAULT

    car_model = YOLO(str(car_path))
    car_result = car_model.predict(str(image_path), imgsz=imgsz, conf=conf, verbose=False)[0]
    car_detections = []
    if car_result.boxes is not None:
        for box, cls_id, score in zip(car_result.boxes.xyxy.tolist(),
                                      car_result.boxes.cls.tolist(),
                                      car_result.boxes.conf.tolist()):
            name = car_model.names[int(cls_id)]
            if name != "car":  # COCO detector filtered to cars only
                continue
            car_detections.append({
                "label": "car", "confidence": round(float(score), 4),
                "bbox_xyxy": [round(float(v), 1) for v in box],
            })

    part_detections, part_status, part_model_info = [], "NOT_TRAINED", None
    part_masks_xy = []
    if part_path.exists():
        part_model = YOLO(str(part_path))
        part_result = part_model.predict(str(image_path), imgsz=imgsz, conf=conf, verbose=False)[0]
        part_status = "TRAINED"
        part_model_info = {"weights": str(part_path), "task": part_model.task,
                           "class_names": list(part_model.names.values())}
        if part_result.masks is not None:
            # Polygon coordinates in original-image space, used only for drawing
            # the model's own masks on the annotated image (never invented).
            part_masks_xy = [poly for poly in part_result.masks.xy]
        if part_result.boxes is not None:
            masks = part_result.masks.data.tolist() if part_result.masks is not None else [None] * len(part_result.boxes)
            for box, cls_id, score, mask in zip(part_result.boxes.xyxy.tolist(),
                                                part_result.boxes.cls.tolist(),
                                                part_result.boxes.conf.tolist(), masks):
                det = {"part": part_model.names[int(cls_id)], "confidence": round(float(score), 4),
                       "bbox_xyxy": [round(float(v), 1) for v in box]}
                if mask is not None:
                    det["mask_present"] = True
                part_detections.append(det)
    else:
        part_model_info = {
            "weights": str(part_path), "status": "MISSING",
            "note": ("No fine-tuned part model found. Train with the Colab notebook and place "
                     "best.pt at models/part_detection/best.pt. yolov8n.pt is NOT a substitute."),
        }

    # Annotated image: car boxes (blue) + part masks (translucent green overlay,
    # drawn from the model's own mask polygons) + part boxes (green).
    # Only model outputs are drawn — nothing is invented.
    import cv2
    annotated = cv2.imread(str(image_path))
    for d in car_detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox_xyxy"]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(annotated, f"car {d['confidence']:.2f}", (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    if part_masks_xy:
        overlay = annotated.copy()
        for poly in part_masks_xy:
            if poly is None or len(poly) < 3:
                continue
            cv2.fillPoly(overlay, [poly.astype(int)], (0, 255, 0))
        annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)
    for d in part_detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox_xyxy"]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{d['part']} {d['confidence']:.2f}", (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    result = {
        "model_info": {
            "car_detector": {"weights": str(car_path), "type": "YOLOv8n COCO pretrained (generic, NOT the project model)",
                             "detects": "car only"},
            "part_model": {"status": part_status, **(part_model_info or {})},
            "make_model_year": "Unknown (no classifier trained yet)",
            "damage_type": None,
            "preprocessing": f"YOLO predict, imgsz={imgsz}, conf={conf}; image read as-is, no manual edits",
        },
        "input_image": str(image_path),
        "car_detections": car_detections,
        "part_detections": part_detections,
        "summary": {
            "cars_detected": len(car_detections),
            "parts_detected": len(part_detections),
            "part_model_trained": part_status == "TRAINED",
        },
    }
    return result, annotated


def main():
    ap = argparse.ArgumentParser(description="Pakistani Car Damage AI — standalone inference (see module docstring).")
    ap.add_argument("--image", required=True, help="Path to a car image (JPG/PNG/WEBP).")
    ap.add_argument("--weights", default=None, help="Part-model weights (default: models/part_detection/best.pt if present).")
    ap.add_argument("--car-weights", default=None, help="Car detector weights (default: yolov8n.pt).")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--output-dir", default="outputs/predict_test", help="Folder for annotated image + JSON.")
    ap.add_argument("--json-out", default=None, help="JSON output path (default: <output-dir>/prediction.json).")
    ap.add_argument("--no-annotated", action="store_true", help="Skip writing the annotated image.")
    args = ap.parse_args()

    result, annotated = run_inference(args.image, part_weights=args.weights,
                                      car_weights=args.car_weights, conf=args.conf, imgsz=args.imgsz)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_annotated:
        import cv2
        ann_path = out_dir / f"annotated_{Path(args.image).stem}.jpg"
        cv2.imwrite(str(ann_path), annotated)
        result["annotated_image"] = str(ann_path)
    json_path = Path(args.json_out) if args.json_out else out_dir / "prediction.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {json_path}" + (f" and {result.get('annotated_image', '')}" if not args.no_annotated else ""))


if __name__ == "__main__":
    sys.exit(main())
