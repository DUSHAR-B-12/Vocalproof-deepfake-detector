# 🛡️ VoiceShield — AI Deepfake Voice Detection

Real-time deepfake voice detection system powered by a **ResNet-style deep learning model** trained on log-mel spectrograms.

Detects AI-generated (fake) voices vs. real human speech with high confidence.

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/AIfakevoicesddetection.git
cd AIfakevoicesddetection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your trained model
# Place best_model.keras inside: data/models/<run_name>/

# 4. Start the server
python run_server.py

# 5. Open the UI
# → http://127.0.0.1:8000/ui
```

---


## 🏗️ Architecture

| Component | Details |
|-----------|---------|
| **Features** | 80-bin log-mel spectrogram, 4s clips, per-sample normalized |
| **Augmentation** | SpecAugment (time + frequency masking) |
| **Model** | ResNet-style CNN: stem → 3 residual stages (32/64/128) → GAP → Dense → Sigmoid |
| **Training** | Binary cross-entropy + label smoothing, Adam optimizer, val AUC checkpointing |
| **Metrics** | AUC, EER, confusion matrix |
| **Backend** | FastAPI + uvicorn |
| **Frontend** | Vanilla HTML/CSS/JS, dark AI theme, drag-and-drop upload |

---

## 🔧 Configuration

All paths are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `VOICESHIELD_DATASET` | `data/dataset` | Dataset root |
| `VOICESHIELD_MODELS` | `data/models` | Model output directory |
| `VOICESHIELD_LOGS` | `data/logs` | TensorBoard logs |
| `VOICESHIELD_MODEL_PATH` | Auto-discovered | Direct path to `.keras` model |
| `VOICESHIELD_UPLOADS` | `data/uploads` | Temp upload directory |

Copy `.env.example` to `.env` and modify as needed.

---

## 📡 API

### `POST /predict`

Upload an audio file → get prediction.

```bash
curl -X POST http://localhost:8000/predict -F "file=@audio.wav"
```

**Response:**
```json
{
  "label": "FAKE",
  "confidence": 0.9787
}
```

### `GET /ui`
Web interface for drag-and-drop detection.

### `GET /docs`
Interactive Swagger API documentation.

---

## 🧪 Ensemble Prediction (CLI)

```bash
python -m inference.ensemble_predict "path/to/audio.wav"
```

Combines `best_model.keras` + `final_model.keras` via probability averaging.

---

## 📊 Training

```bash
# Retrain the model
python -m training.train
```

Requires dataset in `data/dataset/` with `metadata_clean.csv`.

---

## 📦 Requirements

- Python 3.10+
- TensorFlow / Keras
- Librosa
- FastAPI + uvicorn
- NumPy, Pandas, scikit-learn

---

## 📝 License

MIT

---

**Built with ❤️ by VoiceShield AI**
