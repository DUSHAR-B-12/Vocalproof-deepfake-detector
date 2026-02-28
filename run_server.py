"""
VoiceShield — FastAPI Server for Deepfake Voice Detection
Usage: python run_server.py
"""

import os
import sys
import uuid
import shutil
import numpy as np
from pathlib import Path

# Ensure project root is importable
ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from training.feature_extraction import extract_features
from training.spec_augment import SpecAugment
from contextlib import asynccontextmanager

# ── Config (portable paths from training.config) ─────────────────
from training.config import MODEL_DIR, PROJECT_ROOT

def _find_best_model() -> Path:
    """Auto-discover best_model.keras in the latest run directory."""
    # Allow direct override via env var
    env_path = os.environ.get("VOICESHIELD_MODEL_PATH")
    if env_path:
        return Path(env_path)
    # Search for most recent run with best_model.keras
    if MODEL_DIR.exists():
        candidates = sorted(MODEL_DIR.glob("*/best_model.keras"), reverse=True)
        if candidates:
            return candidates[0]
    return MODEL_DIR / "best_model.keras"

MODEL_PATH = _find_best_model()
UPLOAD_DIR = Path(os.environ.get("VOICESHIELD_UPLOADS",
                                  str(PROJECT_ROOT / "data" / "uploads")))
ALLOWED_EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}

# ── Global model ─────────────────────────────────────────────────
model = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    global model
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Loading model from {MODEL_PATH} ...")
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")
    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        custom_objects={"SpecAugment": SpecAugment},
    )
    print(f"[OK]   Model loaded — ready for inference")
    yield
    model = None


# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="VoiceShield API",
    description="Real-time deepfake voice detection",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    label: str
    confidence: float


@app.get("/")
async def root():
    return {"status": "online", "model": MODEL_PATH.name}


@app.get("/ui")
async def serve_ui():
    """Serve the frontend UI."""
    from fastapi.responses import FileResponse
    ui_path = Path(__file__).parent / "frontend" / "index.html"
    if not ui_path.exists():
        raise HTTPException(404, "Frontend not found")
    return FileResponse(str(ui_path), media_type="text/html")


# Mount frontend static files (css, js) at /static
from fastapi.staticfiles import StaticFiles
_frontend_dir = Path(__file__).parent / "frontend"
if _frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported format '{ext}'")

    tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        spec = extract_features(str(tmp_path))
        x = np.expand_dims(spec, axis=0)
        prob = float(model.predict(x, verbose=0).flatten()[0])

        label = "FAKE" if prob >= 0.5 else "REAL"
        confidence = prob if label == "FAKE" else 1.0 - prob
        return PredictionResponse(label=label, confidence=round(confidence, 4))
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "run_server:app",      # import from THIS file, no package needed
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[ROOT],
    )
