# models/ — where trained weights live (READ BEFORE YOU COPY ANYTHING HERE)

## The one rule

**`yolov8n.pt` in the project root is NOT the project model.**
It is Ultralytics' generic COCO pretrained YOLOv8n car detector. It is used
only for whole-car detection in `predict.py` and the API. Never present it
as a model trained on Carparts-Seg, and never present its car-detection
confidence as a part/damage accuracy.

## Where the REAL project model goes (after Colab training)

| File | What it is | How it is produced |
|---|---|---|
| `models/part_detection/best.pt` | Fine-tuned YOLOv8n-seg part-segmentation model (23 Carparts classes) | `colab/Train_CarParts_Seg_YOLOv8n_Seg.ipynb`, Cell 10 export, then copy the exported `best.pt` to exactly this path |
| `models/car_detection/yolov8n.pt` | Optional copy of the generic car detector | Not required — `predict.py` falls back to root `yolov8n.pt` |
| `models/damage_detection/best.pt` | Damage model | **NOT TRAINED YET — future work.** Do not place a renamed copy of any other model here. |
| `models/car_classification/best.pt` | Make/model classifier | **NOT TRAINED YET — future work.** Do not place a renamed copy of any other model here. |

## How inference finds the weights (no code change needed)

`predict.py` and `api/main.py` check for `models/part_detection/best.pt`
on every run:

- **File present** → part model loads, `part_model_status = "TRAINED"`,
  part detections (with real confidences) appear in the output.
- **File missing** → `part_model_status = "NOT_TRAINED"`, `part_detections = []`,
  and the output says plainly that no fine-tuned part model exists.
  Car detection still works. Nothing is invented to fill the gap.

## Distribution (for the evaluator)

Do **not** commit the dataset, and do not rely on committing `best.pt`
(the existing `.gitignore` excludes `models/*/best.pt`). After training:

1. Upload `best.pt` as a **GitHub Release asset** in your repository.
2. Paste the Release download link into `SUBMISSION_README.md` (Links section).
3. The evaluator downloads it, places it at `models/part_detection/best.pt`,
   and runs `python predict.py --image <their image>`.

## Verification after you copy best.pt

```
python predict.py --image uploads/images/honda_civic_cc0.jpg --output-dir outputs/predict_test
```

Check the JSON: `part_model_status` must say `"TRAINED"` and
`model_info.part_model.weights` must end with `models/part_detection/best.pt`.
If it still says `NOT_TRAINED`, the file is in the wrong folder or misnamed.
