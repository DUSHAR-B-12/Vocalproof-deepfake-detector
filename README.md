# 🎤 VoiceShield - Production-Ready Voice Detection System

A complete full-stack application for AI-generated voice detection using MERN stack + FastAPI ML backend.

## 🏗️ Architecture

```
VoiceShield/
├── Frontend (React.js)        → http://localhost:3000
├── Backend (Node.js/Express)  → http://localhost:5000
└── ML Service (FastAPI)       → http://localhost:8000
```

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ and npm
- Python 3.10+
- Git

### 1️⃣ Start ML Service (FastAPI)

```powershell
cd "d:\MINI project\ml-service"
python run_web.py
```

**Expected Output:**
```
Uvicorn running on http://0.0.0.0:8000
```

### 2️⃣ Start Backend Server (Express)

```powershell
cd "d:\MINI project\Backend"
npm install  # (only first time)
node server.js
```

**Expected Output:**
```
🎙️ VoiceShield Backend running on http://0.0.0.0:5000
📡 ML Service expected at http://localhost:8000
```

### 3️⃣ Start Frontend (React)

```powershell
cd "d:\MINI project\Front end"
npm install  # (only first time)
npm run dev
```

**Expected Output:**
```
VITE v5.4.21  ready in XXX ms
➜  Local:   http://localhost:3000/
```

## 🌐 Access the Application

Once all three services are running:
- **Frontend UI:** http://localhost:3000
- **Backend API:** http://localhost:5000/api/audio/health
- **ML Service:** http://localhost:8000/health

## 📋 Features

### Frontend
- ✅ Drag & drop audio file upload
- ✅ Audio preview with player controls  
- ✅ Real-time prediction display
- ✅ Confidence score visualization
- ✅ Scrollable history panel
- ✅ localStorage persistence
- ✅ Responsive design
- ✅ Loading animations

### Backend
- ✅ Express.js REST API
- ✅ Multer file upload handling (50MB limit)
- ✅ CORS enabled for frontend
- ✅ Error handling and validation
- ✅ Temporary file cleanup
- ✅ Health check endpoints

### ML Service (FastAPI)
- ✅ One-Class ML model for voice authenticity
- ✅ MFCC + spectral feature extraction
- ✅ Audio preprocessing
- ✅ JSON REST API
- ✅ Confidence scoring

## 🔄 Data Flow

```
Frontend (React)
    ↓ (file upload)
Backend (Express/Multer)
    ↓ (forward to ML)
ML Service (FastAPI)
    ↓ (feature extraction + inference)
Backend (Express)
    ↓ (add metadata + history)
Frontend (React)
    ↓ (display result + save to localStorage)
```

## 📁 Project Structure

```
d:\MINI project\
├── Frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── AudioUpload.jsx
│   │   │   ├── AudioHistory.jsx
│   │   │   └── ResultDisplay.jsx
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   ├── index.html
│   ├── vite.config.js
│   ├── package.json
│   └── node_modules/
│
├── Backend/
│   ├── server.js
│   ├── routes/
│   │   └── audio.js
│   ├── uploads/  (temp files)
│   ├── .env
│   ├── package.json
│   └── node_modules/
│
└── ml-service/ (FastAPI)
    ├── app.py
    ├── run_web.py
    ├── src/
    │   ├── predict.py
    │   ├── preprocess.py
    │   ├── extract_features.py
    │   └── train_model.py
    ├── models/
    │   └── voice_model.pkl
    ├── data/
    └── requirements.txt
```

## 🛠️ API Endpoints

### Backend Endpoints

**Predict Voice**
```http
POST /api/audio/predict
Content-Type: multipart/form-data

Body: { file: <audio_file> }

Response:
{
  "success": true,
  "prediction": "REAL" | "FAKE",
  "confidence": 85,
  "fileName": "sample.wav",
  "uploadedAt": "2026-02-03T12:00:00Z"
}
```

**Health Check**
```http
GET /api/audio/health

Response:
{
  "backend": "ok",
  "ml_service": "ok",
  "ml_model_loaded": true
}
```

## 🎨 UI Components

### AudioUpload
- Drag & drop zone
- File input selector
- Supported formats display

### ResultDisplay
- Prediction badge (REAL/FAKE)
- Confidence score with progress bar
- File metadata
- Timestamp
- Descriptive message

### AudioHistory
- Scrollable list of past predictions
- Quick access to previous results
- Clear history button
- Selection highlighting

## 🔐 Security Features

- ✅ File type validation (audio only)
- ✅ File size limits (50MB max)
- ✅ CORS protection
- ✅ Temporary file cleanup
- ✅ Error handling without stack traces in production

## 📊 Technology Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 18 + Vite + CSS3 |
| Backend | Node.js + Express + Multer |
| ML | FastAPI + Scikit-learn + Librosa |
| Storage | localStorage (frontend) |
| File Upload | Multer |

## 🐛 Troubleshooting

### ML Service fails to start
```bash
pip install -r requirements.txt
python run_web.py
```

### Backend can't connect to ML Service
- Ensure ML Service is running on port 8000
- Check firewall settings
- Verify environment variable: ML_SERVICE_URL=http://localhost:8000

### Frontend shows "Can't connect"
- Verify Backend is running on port 5000
- Check CORS is enabled
- Clear browser cache (Ctrl+Shift+Delete)

### App.css not found error
- Delete `Frontend/node_modules`
- Run `npm install` again
- Restart `npm run dev`

## 📈 Performance

- Frontend: ~100-200ms render
- Upload: Supports up to 50MB
- Processing: 2-5 seconds per audio file
- History: Stores 50 latest predictions (localStorage)

## 🔄 Development Workflow

1. **Frontend Development**
   ```bash
   cd Frontend
   npm run dev  # Hot reload enabled
   ```

2. **Backend Development**
   ```bash
   cd Backend
   npm install nodemon -g  # Global
   npm run dev  # Auto-restart on changes
   ```

3. **ML Development**
   ```bash
   cd ml-service
   python run_pipeline.py data  # Train model
   python run_web.py            # Start API
   ```

## 📝 Environment Variables

### Backend (.env)
```
PORT=5000
NODE_ENV=development
ML_SERVICE_URL=http://localhost:8000
```

## 🚀 Production Deployment

### Frontend (Vercel/Netlify)
```bash
npm run build
# Upload dist/ folder
```

### Backend (Heroku/Railway)
```bash
npm install
npm start
```

### ML Service (AWS/GCP)
- Use Uvicorn + Gunicorn
- Set PYTHONUNBUFFERED=1
- Ensure model.pkl is available

## 📄 License

MIT License - Free to use and modify

## 👥 Support

For issues or questions, check the deployment terminals and logs.

---
