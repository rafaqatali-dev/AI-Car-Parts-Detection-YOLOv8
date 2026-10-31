# AI Car Parts Detection — YOLOv8n-seg

**Repository (owner-created, public):**
https://github.com/rafaqatali-dev/AI-Car-Parts-Detection-YOLOv8

**What this is:** a car **part segmentation** system. Upload a car photo and
the system detects the car, segments its individual body parts with a trained
YOLOv8n-seg model, and returns part names, confidence scores, and an annotated
image with green segmentation masks.

**What this is NOT (honest scope):** this system does **not** detect damage,
does **not** recognise make/model/year, and does **not** estimate repair or
insurance prices. Those features are future work and are labelled
NOT IMPLEMENTED wherever they appear in the GUI and API responses. Nothing in
this repo should be described as a damage detector.

**Deployment status: NOT DEPLOYED.** There is no public demo URL yet. The
system runs locally (see below). The frontend's `config.js` has no API URL
filled in for that reason — for local testing, type the local API address
into the GUI's API field (see "Run the GUI").

---

## 1. Trained model (included in this repo)

| File | What it is | Size | MD5 |
|---|---|---|---|
| `models/part_detection/best.pt` | **The trained project model** — YOLOv8n-seg, fine-tuned on Carparts-Seg, 23 classes, 25 epochs on a Colab T4 GPU (see `training/`) | 7,045,940 bytes (~6.7 MB) | `c99271ab36b92d655ea03dc5e1f4cb3a` |
| `yolov8n.pt` | Generic Ultralytics COCO YOLOv8n detector (80 classes) used **only** for whole-car bounding boxes. It is **NOT** the project model and must never be described as one. | 6,549,796 bytes (~6.2 MB) | `95a2449609c73cd69a072b09daaff0cc` |

**GitHub file-size check:** both weight files are far below GitHub's
100 MB per-file limit (and below the 50 MB warning threshold), so **no Git
LFS is needed** and the weights are committed directly in this repo.

**Verify the weights after cloning (recommended for automated testing):**

```bash
# Linux / macOS
md5sum models/part_detection/best.pt
# Expected: c99271ab36b92d655ea03dc5e1f4cb3a
# Expected size: 7045940 bytes

# Windows PowerShell
Get-FileHash models\part_detection\best.pt -Algorithm MD5
```

You can also verify through the API: `GET /health` returns
`part_model_status: "TRAINED"`, `part_model_verified: true`,
`part_model_size_bytes: 7045940` and the MD5 above when the correct file is
present. If `best.pt` is missing, the API still starts but reports
`NOT_TRAINED` and returns empty `part_detections` — that is the honest
fallback, not an error to hide.

**If the weights are ever re-trained:** update the size + MD5 in this README
and in `api/weights.py` together, and never claim the old numbers for new
weights.

---

## 2. The 23 part classes (in model order)

These are the original Carparts-Seg class names, preserved unchanged.
The model does **not** output fenders, roof, grille, or side skirts — those
parts are not in this dataset (see `datasets/README.md`).

| ID | Class | ID | Class |
|---|---|---|---|
| 0 | back_bumper | 12 | front_left_light |
| 1 | back_door | 13 | front_light |
| 2 | back_glass | 14 | front_right_door |
| 3 | back_left_door | 15 | front_right_light |
| 4 | back_left_light | 16 | hood |
| 5 | back_light | 17 | left_mirror |
| 6 | back_right_door | 18 | object (catch-all, unreliable — see evaluation notes) |
| 7 | back_right_light | 19 | right_mirror |
| 8 | front_bumper | 20 | tailgate |
| 9 | front_door | 21 | trunk |
| 10 | front_glass | 22 | wheel |
| 11 | front_left_door | | |

The same list, in machine-readable form, is in
`training/carparts-seg.yaml` (template — the dataset itself is not included;
see §6).

---

## 3. Installation

**Requirements:** Python 3.10+ (tested with Python 3.12), pip, and internet
for the one-time install. CPU only — no GPU is needed for inference.
`requirements.txt` pins the exact tested versions, including CPU-only
PyTorch (`torch==2.15.0+cpu`), so no large CUDA download is needed.

```bash
# 1. Clone (owner pushes this folder to the repo URL above first)
git clone https://github.com/rafaqatali-dev/AI-Car-Parts-Detection-YOLOv8.git
cd AI-Car-Parts-Detection-YOLOv8

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

No API keys, accounts, or secrets are needed. Nothing in this repo reads
environment secrets; the only optional environment variable is
`ALLOWED_ORIGINS` (CORS, see §5) and `MODEL_WEIGHTS_URL` (only used if
`best.pt` is missing — see `api/weights.py`).

---

## 4. Inference — three ways (same model, same results)

### A. Command line (simplest, best for automated testing)

```bash
python predict.py --image examples/sample_civic_cc0.jpg --output-dir outputs/predict_test
```

Output: `outputs/predict_test/prediction.json` (cars, parts, confidences,
mask flags) and `outputs/predict_test/annotated_<name>.jpg` (blue box = car,
green translucent overlay = part segmentation mask, green box = part box).
Options: `--conf` (default 0.25), `--imgsz` (default 640),
`--weights` / `--car-weights` (defaults: `models/part_detection/best.pt`
and `yolov8n.pt`), `--no-annotated` to skip the image.

**Expected result for the included sample** (`examples/sample_civic_cc0.jpg`,
a CC0 photo — see §6): 2 cars (0.8856, 0.5680) and 3 parts
(right_mirror 0.2864, back_left_door 0.2854, back_right_door 0.2653, all with
masks). Part confidences on this photo are deliberately weak — it is an
out-of-distribution photo, not a Carparts-style photo. For strong part
results, use a clear front or rear car photo filling most of the frame.
`make_model_year` is "Unknown" and `damage_type` is `null` in every output:
those features are not implemented.

### B. FastAPI backend

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

- `GET /health` — status, car-detector presence, part-model status +
  size + MD5 + verified flag.
- `POST /predict` — multipart form field `file` (JPG/PNG/WEBP, max 20 MB).
  Returns JSON with `car_detections`, `part_detections` (part, confidence,
  bbox, `mask_present`), `annotated_image_base64`, and `model_info`
  (including `damage_type: null` and make/model "Unknown").
- A non-image upload returns HTTP 400.

### C. Bootstrap GUI (frontend)

```bash
# In a second terminal, from the repo root:
python -m http.server 8080 --directory frontend
# Open http://127.0.0.1:8080/index.html
```

In the GUI's API address field, type `http://127.0.0.1:8000` (or set
`API_BASE_URL` in `frontend/config.js` to that address), choose a car photo,
and click Submit. The **Prediction result** box is the live result. The
**Example results** section further down the page is static documentation,
not your result — do not mistake it for a live prediction.

---

## 5. CORS and configuration

- Local testing: the API allows all origins by default (`ALLOWED_ORIGINS`
  unset), which is safe on a laptop because both servers are local.
- If the frontend is ever deployed publicly, set `ALLOWED_ORIGINS` to the
  exact deployed frontend domain (no trailing slash), restart the API, and
  re-test from the deployed page. A browser CORS error then means the domain
  does not exactly match — fix the variable, not the code.
- `frontend/config.js` is the ONLY file the frontend needs edited for a
  deployment (API URL + link URLs). `GITHUB_URL` is already set to this
  repo; report / presentation / YouTube links are empty until the owner
  publishes them — the page shows "link to be added by owner" instead of a
  fake link.

---

## 6. Dataset — source, description, and licensing limitations

**The training dataset is NOT included in this repo, deliberately.**

- **Dataset:** Carparts-Seg (Ultralytics curated set), documented at
  https://docs.ultralytics.com/datasets/segment/carparts-seg
  and downloadable (for training reproduction) from the official Ultralytics
  release: https://github.com/ultralytics/assets/releases/download/v0.0.0/carparts-seg.zip
- **Contents (verified locally):** 3,833 images (640×640) + 3,833 YOLO
  segmentation-polygon label files, split train 3,156 / val 401 / test 276,
  20,374 annotations across the 23 classes in §2. The shipped YAML split
  comments in the archive are wrong (they say 3516/276/401); the corrected
  config used for training is `training/carparts-seg.yaml`.
- **Quality notes (verified):** 135 label files are empty (train 116 /
  val 12 / test 7) — missing annotations, not background; they were
  quarantined in the Colab training copy only. 23 label lines with 9 tokens
  are valid 4-point polygons, not errors. Thin classes: tailgate ~101
  instances, `object` ~12 instances — `object` scores 0.0 mAP and is
  unreliable (see `training/EVALUATION.md`).
- **Licensing limitations (read before reuse):** the original Roboflow
  project (Gianmarco Russo, 2023) states CC BY 4.0, but the Ultralytics
  curated 3,833-image set states NO dataset licence (the "AGPL-3.0" in its
  YAML is the code licence, not a dataset licence). **Commercial rights are
  NOT confirmed.** This repo therefore does NOT redistribute dataset images
  (training, validation, or test), and the dataset must not be re-uploaded
  from this repo. Use here is local/academic with attribution to the sources
  above. Anyone reusing the dataset for commercial training must verify its
  licence independently first.
- **Sample images in this repo (redistribution permitted):**
  `examples/sample_civic_cc0.jpg` is a CC0 photo (Wikimedia, Japanese-spec
  Honda Civic — used for detection testing only, NOT as a Pakistani-market
  identification example), and `examples/annotated_sample_civic_cc0.jpg`
  + `examples/sample_prediction.json` are outputs generated from that CC0
  photo by this project's own model. No dataset image is included anywhere
  in this repo, and no sample image in this repo comes from the training
  dataset.

---

## 7. Training configuration and evaluation results

Full detail: `training/TRAINING_SUMMARY.md` (auto-generated by the training
notebook), `training/EVALUATION.md` (tables + caveats), and
`training/train_colab.ipynb` (the exact 14-cell Colab notebook that was run).

- **Model:** YOLOv8n-seg (segmentation, because the labels are polygons).
- **Hardware:** Google Colab NVIDIA Tesla T4 (~15 GB VRAM).
- **Run:** 2-epoch smoke test (pipeline proof only, weights discarded) then
  a FULL 25-epoch run from fresh `yolov8n-seg.pt`, batch 16, imgsz 640,
  patience 10. The shipped `best.pt` is the 25-epoch FULL run, not the smoke run.
- **Results (from the Colab run, labelled PRELIMINARY — see caveat below):**
  - Validation (401 images): precision 0.6016, recall 0.7921,
    mAP50_box 0.682, mAP50-95_box 0.5699, mAP50_mask 0.6973.
  - Test (276 images): precision 0.6285, recall 0.8188,
    mAP50_box 0.7082, mAP50-95_box 0.5948, mAP50_mask 0.711.
  - Per-class mAP50_box (val / test) is tabulated in
    `training/EVALUATION.md`. Strongest: front_glass, hood, front_bumper.
    Weakest: `object` (0.0 both splits), mirrors, side doors, wheel.
- **PRELIMINARY caveat (important):** these metrics use the dataset's
  shipped splits unchanged. An earlier duplicate review found 191
  cross-split pHash candidate pairs (28 at distance ≤6, 163 at 7–12) that
  were never human-verified pair-by-pair, so the metrics may be optimistic
  and are labelled PRELIMINARY everywhere. No leakage correction was applied
  to the training data. A frozen, de-duplicated test set is future work.
- **Local reproduction (verified):** running this repo's `predict.py` on
  3 Carparts test images reproduced the Colab sample predictions exactly
  (part names + confidences), and the API `/predict` on the same images
  matched `predict.py` exactly. Part segmentation also works on images
  where the generic car detector finds 0 cars — the part model does not
  depend on the car detector succeeding.

---

## 8. Automated testing (for an evaluator)

```bash
# 1. Install (see §3)
pip install -r requirements.txt

# 2. Check the weights are present and correct
md5sum models/part_detection/best.pt
# -> c99271ab36b92d655ea03dc5e1f4cb3a (7045940 bytes)

# 3. CLI test (must report part_model TRAINED)
python predict.py --image examples/sample_civic_cc0.jpg --output-dir outputs/eval_test
# Expect in outputs/eval_test/prediction.json:
#   car_detections: 2 cars (0.8856, 0.5680)
#   part_detections: 3 parts (right_mirror 0.2864, back_left_door 0.2854,
#     back_right_door 0.2653), each mask_present: true
#   model_info.part_model.status: "TRAINED"
#   model_info.damage_type: null (NOT IMPLEMENTED — expected)

# 4. API test
uvicorn api.main:app --host 127.0.0.1 --port 8000 &
curl http://127.0.0.1:8000/health
# Expect: status ok, part_model_status TRAINED, part_model_verified true
curl -X POST -F "file=@examples/sample_civic_cc0.jpg" http://127.0.0.1:8000/predict
# Expect: same cars/parts as step 3, plus annotated_image_base64 (non-empty)

# 5. GUI test (manual)
python -m http.server 8080 --directory frontend
# Open http://127.0.0.1:8080/index.html, set API to http://127.0.0.1:8000,
# upload examples/sample_civic_cc0.jpg, Submit, and confirm the
# Prediction result box shows the same parts/confidences as step 3.
```

If any step reports `NOT_TRAINED` or empty `part_detections`, the
`models/part_detection/best.pt` file is missing or wrong — re-check §1
before concluding the code is broken.

---

## 9. Project structure

```
.
├── README.md                    # this file
├── requirements.txt             # pinned, tested dependencies (CPU)
├── predict.py                   # standalone inference (CLI)
├── yolov8n.pt                   # generic COCO car detector (NOT the project model)
├── api/
│   ├── main.py                  # FastAPI app: GET /health, POST /predict
│   └── weights.py               # startup weight check/download (never overwrites an existing best.pt)
├── frontend/
│   ├── index.html               # Bootstrap GUI
│   ├── app.js                   # upload + render prediction + example results
│   ├── styles.css
│   ├── config.js                # ONLY file to edit for deployment (API URL + links)
│   └── vercel.json              # static frontend config (deployment is NOT done)
├── models/
│   ├── README.md                # weight slots + TRAINED/NOT_TRAINED behaviour
│   └── part_detection/
│       └── best.pt              # TRAINED 23-class YOLOv8n-seg model
├── examples/
│   ├── sample_civic_cc0.jpg     # CC0 test photo (permitted to redistribute)
│   ├── annotated_sample_civic_cc0.jpg  # model output for that photo
│   └── sample_prediction.json   # model output JSON for that photo
├── datasets/
│   └── README.md                # dataset source + licence limitations (NO images included)
└── training/
    ├── train_colab.ipynb        # exact Colab notebook that trained best.pt
    ├── TRAINING_SUMMARY.md      # auto-generated Colab run summary
    ├── EVALUATION.md            # metrics tables + PRELIMINARY caveats
    ├── carparts-seg.yaml        # corrected 23-class config (template; dataset not included)
    └── sample_predictions_colab.json  # Colab sample predictions (numbers only, no images)
```

**Deliberately excluded (see `.gitignore`):** `.venv/`, datasets,
`outputs/`, `uploads/`, caches, `__pycache__/`, `.env` and any secrets,
editor files, and generated private files. This repo contains no passwords,
API keys, tokens, or personal credentials — none are needed to run it.

---

## 10. Pushing this folder to GitHub (owner steps — NOT done by the assistant)

```bash
cd github_submission
git init
git add .
git status   # check: best.pt and yolov8n.pt must be listed; datasets/ must NOT be
git commit -m "AI Car Parts Detection - trained YOLOv8n-seg (23 classes)"
git branch -M main
git remote add origin https://github.com/rafaqatali-dev/AI-Car-Parts-Detection-YOLOv8.git
git push -u origin main
```

Before pushing, the owner should decide whether to include any large
research files (e.g. duplicate-review sheets) — they are intentionally NOT
in this folder. After pushing, verify on github.com that `best.pt` is present
and that no dataset images, `.env` files, or private files were pushed.
