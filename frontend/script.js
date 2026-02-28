/* ═══════════════════════════════════════════════════════════
   VoiceShield — Frontend Logic
   File upload · API integration · Result rendering
   ═══════════════════════════════════════════════════════════ */

const API_URL = "http://127.0.0.1:8000/predict";

// ── DOM Elements ─────────────────────────────────────────
const dropZone      = document.getElementById("dropZone");
const fileInput      = document.getElementById("fileInput");
const fileNameEl     = document.getElementById("fileName");
const audioPreview   = document.getElementById("audioPreview");
const audioPlayer    = document.getElementById("audioPlayer");
const predictBtn     = document.getElementById("predictBtn");
const spinnerWrapper = document.getElementById("spinnerWrapper");
const statusMsg      = document.getElementById("statusMsg");
const resultBox      = document.getElementById("result");
const resultBadge    = document.getElementById("resultBadge");
const resultLabel    = document.getElementById("resultLabel");
const confText       = document.getElementById("confText");
const progressFill   = document.getElementById("progressFill");

let selectedFile = null;

// ── File Handling ────────────────────────────────────────

/** Process a selected file */
function handleFile(file) {
    if (!file) return;

    // Validate extension
    const ext = file.name.split(".").pop().toLowerCase();
    const allowed = ["wav", "flac", "mp3", "m4a", "ogg"];
    if (!allowed.includes(ext)) {
        showStatus(`Invalid file type ".${ext}". Accepted: ${allowed.join(", ")}`, "warn");
        return;
    }

    selectedFile = file;
    hideStatus();
    hideResult();

    // Update UI
    fileNameEl.textContent = file.name;
    dropZone.classList.add("drop-zone--has-file");
    predictBtn.disabled = false;

    // Audio preview
    const url = URL.createObjectURL(file);
    audioPlayer.src = url;
    audioPreview.classList.add("show");
}

// ── Drag & Drop ──────────────────────────────────────────
fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drop-zone--dragover");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drop-zone--dragover");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drop-zone--dragover");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

// ── Prediction ───────────────────────────────────────────
predictBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    // UI → loading state
    predictBtn.disabled = true;
    spinnerWrapper.classList.add("show");
    hideResult();
    hideStatus();

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const res = await fetch(API_URL, {
            method: "POST",
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Server error (${res.status})`);
        }

        const data = await res.json();
        showResult(data.label, data.confidence);

    } catch (err) {
        if (err.message.includes("Failed to fetch")) {
            showStatus("Cannot reach server. Is run_server.py running?", "error");
        } else {
            showStatus(err.message, "error");
        }
    } finally {
        spinnerWrapper.classList.remove("show");
        predictBtn.disabled = false;
    }
});

// ── Result Display ───────────────────────────────────────

/** Show prediction result with animation */
function showResult(label, confidence) {
    const pct = (confidence * 100).toFixed(1);
    const isReal = label === "REAL";

    // Reset classes
    resultBox.className = "result show";
    resultBox.classList.add(isReal ? "result--real" : "result--fake");

    resultBadge.textContent = isReal ? "✓ Verified" : "⚠ Warning";
    resultLabel.textContent = label;
    confText.innerHTML = `${pct}<span style="font-size:0.8rem;opacity:0.7">%</span>`;

    // Animate progress bar after a short delay
    progressFill.style.width = "0%";
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            progressFill.style.width = `${pct}%`;
        });
    });
}

/** Hide result */
function hideResult() {
    resultBox.classList.remove("show", "result--real", "result--fake");
}

// ── Status Messages ──────────────────────────────────────

/** Show status/error message */
function showStatus(msg, type = "error") {
    statusMsg.textContent = msg;
    statusMsg.className = `status-msg show status-msg--${type}`;
}

/** Hide status message */
function hideStatus() {
    statusMsg.classList.remove("show");
}
