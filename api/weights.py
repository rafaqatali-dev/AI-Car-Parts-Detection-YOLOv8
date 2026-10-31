"""Startup weights handling for the public inference API — stdlib only.

Problem it solves: ``models/part_detection/best.pt`` is gitignored on purpose,
so a fresh clone on a host (Hugging Face Space / Render) has NO part model and
the API would honestly report NOT_TRAINED. This module gives the host two safe
ways to get the real trained weights without bundling them in the repo:

1. Upload ``best.pt`` to the host's files yourself (Hugging Face Space file
   upload, or Render's persistent file handling) at ``models/part_detection/best.pt``.
2. Set the ``MODEL_WEIGHTS_URL`` environment variable to the public GitHub
   Release download URL for ``best.pt``. On startup, and ONLY if the file is
   missing, the API downloads it, verifies size + md5, and then uses it.

Safety rules (do not weaken them):
- An existing ``best.pt`` is NEVER overwritten or replaced by a download.
- A download that fails size/md5 verification is deleted and the API still
  starts, reporting NOT_TRAINED — never a partial or wrong model.
- Only http(s) URLs are accepted. No URL means no download attempt.
- The expected values below are for the real 25-epoch Colab FULL run
  (7,045,940 bytes). If the owner ever trains a new model, update BOTH values
  here AND in SUBMISSION_README.md, and re-verify /health before deploying.
"""
import hashlib
import os
import urllib.request
from pathlib import Path

EXPECTED_SIZE_BYTES = 7045940
EXPECTED_MD5 = "c99271ab36b92d655ea03dc5e1f4cb3a"


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_fingerprint(path):
    """Return (size_bytes, md5) for an existing file, else (None, None)."""
    p = Path(path)
    if not p.exists():
        return None, None
    return p.stat().st_size, file_md5(p)


def ensure_part_weights(path, url=None, expected_size=EXPECTED_SIZE_BYTES,
                        expected_md5=EXPECTED_MD5):
    """Return True if a verified part model exists at ``path`` afterwards.

    - Already present: verified against size/md5 and left untouched.
      A present file with a DIFFERENT fingerprint is kept (never deleted),
      but the function returns False so the caller can warn — the API will
      still load it via predict.py, and /health shows its fingerprint.
    - Missing + URL given: download to a temp file, verify, then move in place.
    - Missing + no URL: return False (API reports NOT_TRAINED).
    """
    p = Path(path)
    if p.exists():
        size, md5 = file_fingerprint(p)
        return size == expected_size and md5 == expected_md5
    if not url:
        return False
    if not (url.startswith("https://") or url.startswith("http://")):
        print(f"[weights] Refusing non-http(s) MODEL_WEIGHTS_URL: {url[:60]}")
        return False
    tmp = p.with_suffix(".pt.download")
    try:
        print(f"[weights] Downloading part model from {url} ...")
        urllib.request.urlretrieve(url, tmp)
        size, md5 = file_fingerprint(tmp)
        if size != expected_size or md5 != expected_md5:
            print(f"[weights] Verification FAILED (size={size}, md5={md5}); "
                  "deleting download, keeping NOT_TRAINED state.")
            tmp.unlink(missing_ok=True)
            return False
        p.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, p)
        print(f"[weights] Verified part model saved to {p} "
              f"(size={size}, md5={md5}).")
        return True
    except Exception as exc:  # host starts anyway; /health shows NOT_TRAINED
        print(f"[weights] Download failed ({exc}); API will report NOT_TRAINED.")
        tmp.unlink(missing_ok=True)
        return False
