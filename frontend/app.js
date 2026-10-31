document.addEventListener("DOMContentLoaded", () => {
  const apiInput = document.getElementById("apiUrl");
  const fileInput = document.getElementById("fileInput");
  const preview = document.getElementById("preview");
  const btn = document.getElementById("predictBtn");
  const status = document.getElementById("status");
  const errorBox = document.getElementById("error");
  const resultBox = document.getElementById("result");

  // Fill links from config.js; empty values stay as an honest placeholder, never a fake URL.
  const setLink = (id, url) => {
    const el = document.getElementById(id);
    if (url) { el.href = url; el.textContent = url; } else { el.removeAttribute("href"); }
  };
  setLink("githubLink", CONFIG.GITHUB_URL);
  setLink("reportLink", CONFIG.REPORT_URL);
  setLink("presentationLink", CONFIG.PRESENTATION_URL);
  setLink("youtubeLink", CONFIG.YOUTUBE_URL);
  if (CONFIG.API_BASE_URL) apiInput.value = CONFIG.API_BASE_URL;

  fileInput.addEventListener("change", () => {
    const f = fileInput.files[0];
    if (f) preview.src = URL.createObjectURL(f);
  });

  function showError(msg) {
    errorBox.hidden = false; errorBox.textContent = msg;
    resultBox.hidden = true; status.textContent = "";
  }

  btn.addEventListener("click", async () => {
    const base = (apiInput.value || CONFIG.API_BASE_URL || "").replace(/\/$/, "");
    const file = fileInput.files[0];
    if (!base) return showError("API base URL is not set yet. Edit frontend/config.js (see DEPLOYMENT_GUIDE.md) or type the API URL above.");
    if (!file) return showError("Please choose an image first (JPG, PNG, or WEBP, max 20 MB).");
    if (file.size > 20 * 1024 * 1024) return showError("File too large: max 20 MB.");
    errorBox.hidden = true; resultBox.hidden = true;
    status.textContent = "Sending to the model… (first request on a free host can take 30–60 seconds while it wakes up.)";
    const form = new FormData(); form.append("file", file);
    let resp;
    try {
      resp = await fetch(base + "/predict", { method: "POST", body: form });
    } catch (e) {
      return showError("Could not reach the inference API. The API may be asleep or the URL may be wrong. No result is shown instead of a fake one. Error: " + e.message);
    }
    if (!resp.ok) {
      const t = await resp.text();
      return showError("API returned an error (" + resp.status + "): " + t);
    }
    const data = await resp.json();
    status.textContent = "";
    resultBox.hidden = false;
    document.getElementById("summary").textContent =
      `Cars detected: ${data.summary.cars_detected} | Parts detected: ${data.summary.parts_detected} | Part model trained: ${data.summary.part_model_trained}`;
    document.getElementById("notImplemented").textContent =
      `Damage detection: NOT IMPLEMENTED (model returns damage_type = null). ` +
      `Make / model: ${data.model_info.make_model_year || "Unknown (no classifier trained yet)"}. ` +
      `Insurance pricing: NOT IMPLEMENTED.`;
    document.getElementById("partStatus").textContent =
      data.model_info.part_model.status === "TRAINED"
        ? "Part model: TRAINED (fine-tuned YOLOv8n-seg, 23 classes). Masks below are the model's own segmentation masks."
        : "Part model: NOT TRAINED yet — part list is empty on purpose. Train with the Colab notebook and place best.pt at models/part_detection/best.pt.";
    const fill = (tableId, rows, labelKey, withMask) => {
      const table = document.getElementById(tableId);
      table.querySelectorAll("tr:not(:first-child)").forEach(r => r.remove());
      rows.forEach(r => {
        const tr = document.createElement("tr");
        tr.innerHTML = withMask
          ? `<td>${r[labelKey]}</td><td>${r.confidence}</td><td>${r.mask_present ? "Yes" : "No"}</td><td>${r.bbox_xyxy.join(", ")}</td>`
          : `<td>${r[labelKey]}</td><td>${r.confidence}</td><td>${r.bbox_xyxy.join(", ")}</td>`;
        table.appendChild(tr);
      });
      if (!rows.length) {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td colspan="${withMask ? 4 : 3}">None detected.</td>`;
        table.appendChild(tr);
      }
    };
    fill("carTable", data.car_detections, "label", false);
    fill("partTable", data.part_detections, "part", true);
    if (data.annotated_image_base64) {
      document.getElementById("annotated").src = "data:image/jpeg;base64," + data.annotated_image_base64;
    }
  });
});
