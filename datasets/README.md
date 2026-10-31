# Dataset — source and licensing limitations

**No dataset images are included in this repository. This is deliberate,
not an omission.**

## Source

- **Dataset:** Carparts-Seg (Ultralytics curated set)
- **Documentation:** https://docs.ultralytics.com/datasets/segment/carparts-seg
- **Download (for training reproduction only):**
  https://github.com/ultralytics/assets/releases/download/v0.0.0/carparts-seg.zip
  (134 MB ZIP; 3,833 images at 640×640 + 3,833 YOLO segmentation-polygon
  label files; split train 3,156 / val 401 / test 276; 20,374 annotations;
  23 classes — see `../training/carparts-seg.yaml` and the main README §2.)
- **Original project:** Gianmarco Russo, "car-seg", 2023 (Roboflow).

## Why the images are not here

1. **Licence is not confirmed for the curated set.** The original Roboflow
   project states CC BY 4.0, but the Ultralytics curated 3,833-image set
   states NO dataset licence (the "AGPL-3.0" in its YAML is the code
   licence, not a dataset licence). Commercial rights are NOT confirmed.
2. Re-uploading the images from this repo could therefore infringe the
   dataset owner's rights, even if the original project page says CC BY 4.0.
3. The safe, honest route is: link to the official source (above) and let
   each user download it under whatever terms apply to them, after checking
   the licence themselves.

## Quality notes (verified on the local copy used for training)

- 135 label files are empty (train 116 / val 12 / test 7): missing
  annotations, not background. They were quarantined in the Colab training
  copy only; the originals were not modified.
- 23 label lines with 9 tokens are VALID 4-point polygons, not errors.
  A label line is bad only if it has fewer than 7 tokens, an odd coordinate
  count, or coordinates outside [0, 1] — under that rule, bad lines = 0.
- Thin classes: tailgate ~101 instances, `object` ~12 instances.
  The `object` class scores 0.0 mAP and should be treated as unreliable.
- The shipped YAML split comments in the archive are WRONG
  (they say train 3516 / val 276 / test 401). The corrected config used
  for training is `../training/carparts-seg.yaml`
  (train 3156 / val 401 / test 276).

## What an evaluator should do instead of expecting dataset images here

- Run the included CC0 sample: `examples/sample_civic_cc0.jpg`
  (see the main README §4 and §8 for expected outputs).
- Or run the system on their own car photo — no dataset access is needed
  for inference, because the trained weights ARE included in this repo.
- To reproduce training, download the dataset from the official link
  above, place it as the training notebook expects, and run
  `training/train_colab.ipynb`. Check the dataset licence yourself before
  any commercial use.

## Sample images that ARE included (and why they are permitted)

- `examples/sample_civic_cc0.jpg` — a CC0 photo (public domain dedication),
  used for detection testing only. It is NOT a Pakistani-market car
  identification example.
- `examples/annotated_sample_civic_cc0.jpg` and
  `examples/sample_prediction.json` — outputs generated from that CC0 photo
  by this project's own trained model. They contain no dataset imagery.
