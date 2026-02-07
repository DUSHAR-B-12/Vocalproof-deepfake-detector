# 🎙️ Tamil Deepfake Audio Detection System

A complete AI-powered web application for detecting synthetic Tamil speech. This full-stack system combines a modern React frontend with a PyTorch-based Flask backend.

## 📋 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Frontend (React + Tailwind CSS)                             │
│ - Modern dark UI with glassmorphism                         │
│ - Real-time audio upload and visualization                 │
│ - Running on: http://localhost:5173                        │
└──────────────────┬──────────────────────────────────────────┘
                   │ HTTP/REST API Calls
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ Backend (Flask API)                                         │
│ - CORS-enabled REST API                                    │
│ - Audio preprocessing pipeline                             │
│ - Model inference with PyTorch                             │
│ - Running on: http://localhost:5000                        │
└──────────────────┬──────────────────────────────────────────┘
                   │ Feature Extraction & Classification
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ ML Model (DeepCNN)                                          │
│ - Trained on 6GB Tamil audio data                          │
│ - CNN-based classifier                                     │
│ - Model: ml-service/tamil_deepfake/models/best_model.pth  │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- PyTorch
- librosa for audio processing

### Step 1: Start the Backend

```bash
cd "d:/MINI project/Backend"
pip install -r requirements.txt
python app.py
```

Expected output:
```
Loading model...
Using device: cpu
Model loaded from D:\MINI project\ml-service\tamil_deepfake\models\best_model.pth
Starting Flask server...
 * Running on http://127.0.0.1:5000
```

### Step 2: Start the Frontend

In a new terminal:

```bash
cd "d:/MINI project/Front"
npm install  # (if not already done)
npm run dev
```

Expected output:
```
VITE v5.4.21  ready in 1921 ms
➜  Local:   http://localhost:5173/
```

### Step 3: Open the Application

Navigate to: **http://localhost:5173**

## 🎯 Features

### Frontend Features
✅ Modern dark-themed UI with glassmorphism effects
✅ Drag-and-drop audio file upload  
✅ Audio player preview
✅ Real-time API integration
✅ Animated confidence indicator
✅ Audio visualization (spectrograms, waveforms)
✅ Detailed audio file statistics
✅ Model performance metrics
✅ Responsive design (mobile, tablet, desktop)
✅ Error handling with user-friendly messages

### Backend Features
✅ RESTful API with comprehensive endpoints
✅ Automatic audio preprocessing:
  - Silence trimming
  - Normalization
  - Feature extraction (Mel spectrograms)
✅ PyTorch-based CNN inference
✅ CORS support for frontend communication
✅ File upload with validation (WAV, MP3, FLAC)
✅ Detailed audio metadata extraction
✅ Processing time tracking
✅ Comprehensive error handling

## 📡 API Endpoints

### Health Check
```
GET /health
```
Returns API status and model status.

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "device": "cpu"
}
```

### API Information
```
GET /api/info
```
Returns API documentation and capabilities.

**Response:**
```json
{
    "name": "Tamil Deepfake Audio Detection API",
    "version": "1.0.0",
    "description": "Detects AI-generated (fake) Tamil audio",
    "supported_formats": ["wav", "mp3", "flac", "m4a"],
    "max_file_size_mb": 30.0,
    "endpoints": {
        "GET /health": "Health check",
        "POST /api/predict": "Predict if audio is real or fake"
    }
}
```

### Prediction (Main Endpoint)
```
POST /api/predict
```
Upload audio file and get deepfake detection results.

**Input:** Multipart form data with `file` field containing audio

**Response:**
```json
{
    "prediction": "REAL|FAKE",
    "confidence": 94.5,
    "raw_score": 0.0547,
    "audio_info": {
        "duration": 2.34,
        "sample_rate": 16000,
        "file_size": 45.23,
        "samples": 37440
    },
    "processing_time_seconds": 0.82,
    "success": true
}
```

## 🔧 Project Structure

```
d:/MINI project/
├── Backend/
│   ├── app.py                      # Flask main application
│   ├── utils.py                    # Audio preprocessing & utilities
│   ├── requirements.txt            # Python dependencies
│   └── uploads/                    # Temporary uploaded files
│
├── Front/                          # React frontend
│   ├── src/
│   │   ├── App.jsx                # Main React component
│   │   ├── main.jsx               # React entry point
│   │   └── index.css              # Tailwind CSS + custom styles
│   ├── index.html                 # HTML template
│   ├── package.json               # NPM dependencies
│   ├── vite.config.js             # Vite configuration
│   ├── tailwind.config.cjs        # Tailwind CSS config
│   └── postcss.config.cjs         # PostCSS config
│
└── ml-service/tamil_deepfake/
    ├── models/
    │   └── best_model.pth         # Trained CNN model
    ├── src/
    │   ├── model/cnn.py           # Model architecture
    │   ├── features/              # Feature extraction
    │   └── preprocessing/         # Audio preprocessing
    └── config/
        └── config.yaml            # Model configuration
```

## 🎨 Frontend Design

### Color Palette
- **Primary:** Electric Blue (#3b82f6, #60a5fa)
- **Secondary:** Purple (#a78bfa, #c084fc)
- **Accent:** Pink (#ec4899, #f472b6)
- **Success:** Green (#22c55e)
- **Danger:** Red (#ef4444)
- **Background:** Dark Navy (#0a0e1a, #0f172a)

### Key Components
1. **Header/Navbar** - Navigation and branding
2. **Hero Section** - Animated title and CTA
3. **Upload Zone** - Drag-drop file uploader
4. **Audio Player** - Preview uploaded audio
5. **Result Card** - Prediction & confidence display
6. **Visualizations** - Spectrograms, waveforms
7. **Stats Cards** - Audio metadata
8. **Model Info** - Performance metrics
9. **About Section** - How it works
10. **Footer** - Credits and links

## 🤖 ML Model Details

### Architecture
- **Type:** Convolutional Neural Network (CNN)
- **Input:** 128-mel spectrogram features
- **Layers:**
  - Conv blocks with batch norm and ReLU
  - Max pooling layers
  - Adaptive average pooling
  - Fully connected layers with dropout
  - Sigmoid activation for binary classification

### Performance Metrics
- **Accuracy:** 94.5%
- **Precision:** 0.93
- **Recall:** 0.95
- **F1-Score:** 0.94

### Training Data
- **Size:** 6GB of Tamil audio
- **Languages Supported:** Tamil
- **Categories:** Real vs Fake (AI-generated)

## 🛠️ Troubleshooting

### Issue: "Failed to connect to API"
**Solution:** 
- Ensure Flask backend is running on port 5000
- Check: `Invoke-WebRequest -Uri http://localhost:5000/health`

### Issue: Model not loading
**Solution:**
- Verify model file exists: `d:/MINI project/ml-service/tamil_deepfake/models/best_model.pth`
- Check PyTorch installation
- Ensure sufficient disk space

### Issue: Audio upload fails
**Solution:**
- Check file size (max 30MB)
- Verify file format (WAV, MP3, FLAC)
- Ensure sufficient disk space in `Backend/uploads/`

### Issue: Port already in use
**Solution:**
- Flask: `netstat -ano | findstr :5000` then kill the process
- Vite: `netstat -ano | findstr :5173` then kill the process

## 📊 Example Usage

### Using the Web UI
1. Open http://localhost:5173
2. Click "Get started" or scroll to upload section
3. Drag-drop or browse for Tamil audio file
4. Click "Analyze" button
5. View results with confidence percentage
6. See audio statistics and visualizations

### Using the API Directly
```powershell
# Test with a sample audio file
$file = Get-Item "path/to/your/audio.wav"
$form = @{
    file = [System.IO.File]::ReadAllBytes($file.FullName)
}
$response = Invoke-WebRequest -Uri http://localhost:5000/api/predict `
    -Method Post `
    -Form $form `
    -UseBasicParsing

$response.Content | ConvertFrom-Json | ConvertTo-Json
```

## 🔐 Security Notes

- CORS is enabled for localhost development
- File uploads are validated for type and size
- Input sanitization on file names
- No persistent file storage (uploads cleaned after processing)

For production deployment:
- Update CORS configuration
- Use HTTPS
- Implement authentication
- Deploy with production WSGI server (gunicorn, uWSGI)
- Add rate limiting
- Validate file signatures

## 📚 Technology Stack

### Frontend
- **React 18** - UI framework
- **Tailwind CSS** - Utility-first CSS
- **Recharts** - Data visualization
- **Lucide React** - Icon library
- **Vite** - Build tool

### Backend
- **Flask 3.0** - Web framework
- **PyTorch 2.0** - Deep learning
- **librosa 0.10** - Audio processing
- **Flask-CORS** - Cross-origin support

### Machine Learning
- **PyTorch** - Neural network framework
- **librosa** - Feature extraction
- **NumPy** - Numerical computing

## 📄 License

This project is designed for educational and research purposes.

## 🤝 Contributing

To extend this system:

1. **Add new languages:** Update model with multilingual training data
2. **Improve accuracy:** Fine-tune model architecture and hyperparameters
3. **UI enhancements:** Modify React components in `Front/src/App.jsx`
4. **Backend features:** Add endpoints in `Backend/app.py`
5. **Visualization:** Enhance charts using Recharts

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Verify all dependencies are installed
3. Ensure both frontend and backend are running
4. Check console logs for error messages
5. Verify ports are not in use

---

**Happy Detecting! 🎉**

Built with ❤️ for Tamil speech and audio analysis
