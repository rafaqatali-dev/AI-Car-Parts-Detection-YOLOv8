# Evaluation — trained YOLOv8n-seg part model

**Source of every number below:** the auto-generated Colab training summary
(`TRAINING_SUMMARY.md`, same folder), produced by `train_colab.ipynb` after
the FULL 25-epoch run. No number here was edited, rounded beyond the
summary's own precision, or estimated. Where the summary did not report a
metric, it is marked NOT REPORTED — it is not filled in by guesswork.

## Run identification

- **Model:** YOLOv8n-seg (segmentation), fine-tuned from `yolov8n-seg.pt`
- **Hardware:** Google Colab NVIDIA Tesla T4 (~15 GB VRAM)
- **Run type:** FULL (25 epochs; a separate 2-epoch smoke test proved the
  pipeline first and its weights were discarded, not shipped)
- **Settings:** batch 16, imgsz 640, patience 10
- **Shipped weights:** `models/part_detection/best.pt`
  (7,045,940 bytes, md5 `c99271ab36b92d655ea03dc5e1f4cb3a`)
- **Evaluation status: PRELIMINARY** — see the caveat at the end. Every
  metric in this file inherits that label.

## Overall metrics

| Split | Images | Precision | Recall | mAP50_box | mAP50-95_box | mAP50_mask | mAP50-95_mask |
|---|---|---|---|---|---|---|---|
| Validation | 401 | 0.6016 | 0.7921 | 0.682 | 0.5699 | 0.6973 | NOT REPORTED |
| Test | 276 | 0.6285 | 0.8188 | 0.7082 | 0.5948 | 0.711 | NOT REPORTED |

## Per-class mAP50_box (in class-ID order, §2 of the main README)

| ID | Class | Val | Test |
|---|---|---|---|
| 0 | back_bumper | 0.7646 | 0.7973 |
| 1 | back_door | 0.8301 | 0.8246 |
| 2 | back_glass | 0.8298 | 0.7949 |
| 3 | back_left_door | 0.6347 | 0.6522 |
| 4 | back_left_light | 0.3708 | 0.3536 |
| 5 | back_light | 0.6700 | 0.6554 |
| 6 | back_right_door | 0.3703 | 0.4024 |
| 7 | back_right_light | 0.4986 | 0.3785 |
| 8 | front_bumper | 0.9108 | 0.8751 |
| 9 | front_door | 0.8366 | 0.8397 |
| 10 | front_glass | 0.9138 | 0.9154 |
| 11 | front_left_door | 0.5299 | 0.6173 |
| 12 | front_left_light | 0.3891 | 0.5016 |
| 13 | front_light | 0.7272 | 0.7203 |
| 14 | front_right_door | 0.4241 | 0.3671 |
| 15 | front_right_light | 0.4468 | 0.3362 |
| 16 | hood | 0.8953 | 0.8950 |
| 17 | left_mirror | 0.3271 | 0.4061 |
| 18 | object | 0.0000 | 0.0000 |
| 19 | right_mirror | 0.4033 | 0.3572 |
| 20 | tailgate | 0.5544 | 0.8584 |
| 21 | trunk | 0.4777 | 0.7300 |
| 22 | wheel | 0.3029 | 0.4009 |

**Reading guide:** large, high-contrast parts (front_glass, hood,
front_bumper, doors, glass) score highest. Small parts (mirrors, side
lights, wheel) and the catch-all `object` class score lowest. `object`
(~12 training instances) scores 0.0 on both splits and must not be relied
on — it is a dataset catch-all, not a real part. Tailgate's val/test gap
(0.5544 vs 0.8584) reflects its small instance count (~101), not a
measurement error.

## Local reproduction (verified, not from Colab)

Running this repo's `predict.py` on 3 Carparts test images reproduced the
Colab sample predictions exactly (same part names and confidences):

- Rear view 1: back_glass 0.9061, back_light 0.8811, back_bumper 0.8797,
  back_light 0.8604 — generic car detector found 0 cars, part model still
  found 4 parts.
- Rear view 2: back_bumper 0.9390, back_light 0.9131, back_glass 0.9048,
  back_light 0.8076, back_light 0.3833 — generic car detector found 0 cars.
- Front view: front_glass 0.9470, hood 0.9146, front_bumper 0.7750,
  front_light 0.7733, front_light 0.5810, right_mirror 0.2681.

The API `/predict` on the front-view image matched `predict.py` exactly.
The full Colab sample predictions (numbers only, no images) are in
`sample_predictions_colab.json` in this folder.

**Established fact:** part segmentation does NOT depend on the generic
car detector succeeding. On the two rear-view test images above, the car
detector found 0 cars while the part model still found 4–5 parts.

## PRELIMINARY caveat (read before citing these numbers)

These metrics use the dataset's shipped splits unchanged. An earlier
duplicate review found 191 cross-split perceptual-hash candidate pairs
(28 at Hamming distance ≤6, 163 at 7–12) that were never verified
pair-by-pair by a human, so some near-duplicate images may sit in both
training and test splits and the metrics may be optimistic. No leakage
correction was applied before training. Treat every number above as
PRELIMINARY until a frozen, de-duplicated test set is built and the model
is re-evaluated on it. Do not present these numbers as final, verified
accuracy, and do not compare them against other models' final numbers
without stating this caveat.
