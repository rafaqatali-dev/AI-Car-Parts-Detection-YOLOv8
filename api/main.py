"""FastAPI inference API for Pakistani Car Damage AI — submission version.

Run locally (from the project root, after pip install -r api/requirements.txt):
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health   -> model status WITHOUT running inference (fast, for deployment checks)
    POST /predict  -> multipart image upload -> JSON + base64 annotated image

Honesty rules (same as predict.py, which this API reuses):
- Car detection comes from generic COCO yolov8n.pt and is labelled as such.
- Part segmentation runs ONLY if models/part_detection/best.pt exists (Colab output).
  Otherwise part_detections is [] and part_model_status is "NOT_TRAINED".
- If the API is down, the frontend must show an error — never a fabricated result.
- No API keys or secrets are used or required. Do not add any here.

CORS: set ALLOWED_ORIGINS (comma-separated) to your Vercel domain in production,
e.g. ALLOWED_ORIGINS=https://your-site.vercel.app . The default "*" is for local
demo only and must be narrowed before public deployment (see DEPLOYMENT_GUIDE.md).
Max upload: 20 MB, images only (JPG/PNG/WEBP). Larger/corrupt files are rejected
with HTTP 400, never processed.

Weights on a fresh host: models/part_detection/best.pt is gitignored, so a
fresh clone has no part model. On startup the API runs api/weights.py:
if best.pt is already present it is left untouched; if it is missing AND
MODEL_WEIGHTS_URL is set (public GitHub Release download URL), the file is
downloaded and verified by size + md5 before use. A failed download leaves
the API running with part_model_status NOT_TRAINED — never a wrong model.
"""
import base64
import io
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from predict import PART_WEIGHTS_DEFAULT, find_car_weights, run_inference  # noqa: E402
from api.weights import EXPECTED_MD5, EXPECTED_SIZE_BYTES, ensure_part_weights, file_fingerprint  # noqa: E402

MAX_BYTES = 20 * 1024 * 1024
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app):
    # Startup weights check only: never overwrites an existing best.pt.
    ensure_part_weights(PART_WEIGHTS_DEFAULT, os.environ.get("MODEL_WEIGHTS_URL"))
    yield


app = FastAPI(title="Pakistani Car Damage AI — Inference API",
              description="Real YOLO inference only. See predict.py docstring for honesty rules.",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["*"])


@app.get("/health")
def health():
    car = find_car_weights(None)
    size, md5 = file_fingerprint(PART_WEIGHTS_DEFAULT)
    return {
        "status": "ok",
        "car_detector_weights_present": car is not None,
        "car_detector_weights": str(car) if car else None,
        "car_detector_note": "Generic COCO YOLOv8n — NOT the project model.",
        "part_model_status": "TRAINED" if PART_WEIGHTS_DEFAULT.exists() else "NOT_TRAINED",
        "part_model_weights": str(PART_WEIGHTS_DEFAULT),
        "part_model_size_bytes": size,
        "part_model_md5": md5,
        "part_model_verified": (size == EXPECTED_SIZE_BYTES and md5 == EXPECTED_MD5),
        "part_model_note": ("TRAINED means the fine-tuned 23-class YOLOv8n-seg best.pt "
                            "is present. verified=true means its size + md5 match the "
                            "25-epoch Colab FULL run. If NOT_TRAINED, /predict still "
                            "returns real car detections with part_detections = []."),
        "make_model_year": "Unknown (no classifier trained yet)",
        "damage_type": None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="File too large: max 20 MB.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file (JPG, PNG, or WEBP).")
    try:
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Corrupt or unsupported image file.")
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / f"upload{suffix}"
        tmp_path.write_bytes(data)
        try:
            result, annotated = run_inference(tmp_path)
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
        import cv2
        ok, buf = cv2.imencode(".jpg", annotated)
        result["annotated_image_base64"] = base64.b64encode(buf.tobytes()).decode() if ok else None
        result["annotated_image_note"] = "Blue boxes = car (COCO detector). Green translucent overlays = part segmentation masks from the trained model; green boxes = part boxes."
        return result
