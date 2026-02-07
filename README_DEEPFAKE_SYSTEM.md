# 🎙️ Tamil Deepfake Audio Detection System

A production-ready full-stack web application for detecting AI-generated (deepfake) Tamil audio. This integrated system combines a modern React web interface with a PyTorch-based Flask API backend.

## 🌟 Features at a Glance

### 🎨 Frontend
- **Modern Dark UI** with glassmorphism effects and animated gradients
- **Drag-and-Drop Upload** with visual feedback
- **Real-Time Audio Preview** with built-in player
- **Animated Results Display** with confidence indicators
- **Audio Visualizations** - spectrograms and waveforms
- **Detailed Metadata** - duration, sample rate, file size
- **Responsive Design** - works on mobile, tablet, desktop
- **Error Handling** - user-friendly error messages

### 🤖 Backend & ML
- **REST API** with comprehensive endpoints
- **CNN-Based Classification** trained on 6GB Tamil audio data
- **Real-Time Processing** - 500ms-2s per audio file
- **Audio Preprocessing Pipeline** - silence trimming, normalization, feature extraction
- **CORS Support** for seamless frontend-backend communication
- **Production-Ready** Flask application with error handling

## 📊 System Performance

| Metric | Value |
|--------|-------|
| **Model Accuracy** | 94.5% |
| **Precision** | 0.93 |
| **Recall** | 0.95 |
| **F1-Score** | 0.94 |
| **Processing Time** | 0.5-2.0s |
| **Supported Formats** | WAV, MP3, FLAC, M4A |
| **Max File Size** | 30 MB |

## 🚀 Quick Start (5 minutes)

### Prerequisites
```bash
# Required software
- Python 3.8+
- Node.js 16+
- Git (optional)
```

### Option 1: Run Both Servers Together (Easiest)

**Windows - Using BAT file:**
```bash
cd d:\MINI project
START_ALL.bat
```

**Windows - Using PowerShell:**
```bash
cd d:\MINI project
powershell -ExecutionPolicy Bypass -File START_ALL.ps1
```

This opens two terminals - one for backend, one for frontend.

### Option 2: Run Servers Separately

**Terminal 1 - Backend:**
```bash
cd d:\MINI project\Backend
pip install -r requirements.txt
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd d:\MINI project\Front
npm install  # (first time only)
npm run dev
```

### Access the Application
When both servers are running, open your browser:

🌐 **http://localhost:5173**

## 📝 How to Use

1. **Open** http://localhost:5173
2. **Upload** a Tamil audio file (WAV, MP3, FLAC)
3. **Click** "Analyze" button
4. **View** results with confidence percentage
5. **Check** audio statistics and visualizations

## 🏗️ Project Structure

```
d:/MINI project/
│
├── 📁 Backend/                    # Flask API Server
│   ├── app.py                    # Main Flask application
│   ├── utils.py                  # Audio preprocessing utilities
│   ├── requirements.txt          # Python dependencies
│   └── uploads/                  # Temp file storage
│
├── 📁 Front/                     # React Frontend
│   ├── src/
│   │   ├── App.jsx              # Main React component (single file)
│   │   ├── main.jsx             # React entry point
│   │   └── index.css            # Tailwind CSS + custom styles
│   ├── index.html               # HTML template
│   ├── package.json             # NPM dependencies
│   ├── vite.config.js           # Vite build config
│   ├── tailwind.config.cjs      # Tailwind CSS config
│   └── postcss.config.cjs       # PostCSS config
│
├── 📁 ml-service/tamil_deepfake/
│   ├── models/
│   │   └── best_model.pth       # Trained CNN model (50MB)
│   ├── src/
│   │   ├── model/cnn.py         # CNN architecture
│   │   └── features/            # Feature extraction code
│   └── data/                    # Training dataset info
│
├── 📄 SETUP_GUIDE.md            # Complete setup instructions
├── 📄 TECHNICAL_DOCUMENTATION.md # Architecture & implementation details
├── 📄 START_ALL.bat             # Windows batch launcher
└── 📄 START_ALL.ps1             # PowerShell launcher
```

## 🔌 API Endpoints

### Health Check
```
GET /health
```
Check if API is running and model is loaded.

### API Information
```
GET /api/info
```
Get API documentation and capabilities.

### Main Prediction
```
POST /api/predict
```
Upload audio and get deepfake detection results.

**Example using PowerShell:**
```powershell
$file = Get-Item "your-audio.wav"
$form = @{ file = [System.IO.File]::ReadAllBytes($file.FullName) }
$response = Invoke-WebRequest -Uri http://localhost:5000/api/predict `
    -Method Post -Form $form -UseBasicParsing
$response.Content | ConvertFrom-Json
```

**Response Example:**
```json
{
    "prediction": "REAL",
    "confidence": 92.3,
    "audio_info": {
        "duration": 2.34,
        "sample_rate": 16000,
        "file_size": 45.23
    },
    "processing_time_seconds": 0.82,
    "success": true
}
```

## 🛠️ Technology Stack

### Frontend
- **React 18** - UI framework
- **Tailwind CSS** - Utility styling
- **Recharts** - Data visualizations
- **Lucide Icons** - Icon library
- **Vite** - Build tool (lightning fast)

### Backend
- **Flask 3.0** - Web framework
- **PyTorch 2.0** - Deep learning
- **librosa 0.10** - Audio processing
- **Flask-CORS** - Cross-origin support

### Machine Learning
- **Architecture:** Convolutional Neural Network (CNN)
- **Input:** 128-feature mel spectrograms
- **Training Data:** 6GB of Tamil audio
- **Classes:** Real vs Fake (AI-generated)

## 🎨 Design Highlights

### Color Scheme
- **Electric Blue** #3b82f6 - Primary accent
- **Purple** #a78bfa - Secondary gradient
- **Pink** #ec4899 - Highlight color
- **Dark Navy** #0a0e1a - Background

### Visual Effects
- Animated gradient text on heading
- Glassmorphism with backdrop blur
- Smooth transitions and hover effects
- Circular animated confidence indicator
- Floating status indicators

## ⚙️ Configuration

### Frontend Configuration
Create `Front/.env.local`:
```
VITE_API_URL=http://localhost:5000
```

For production:
```
VITE_API_URL=https://your-api-domain.com
```

### Backend Configuration
Set environment variables:
```
FLASK_ENV=development
DEVICE=cpu (or 'cuda' for GPU)
```

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check if port 5000 is in use
netstat -ano | findstr :5000

# Kill the process using the port
taskkill /PID <PID> /F

# Then retry
python app.py
```

### Frontend won't load
```bash
# Check if port 5173 is in use
netstat -ano | findstr :5173

# Clear npm cache
npm cache clean --force

# Reinstall dependencies
npm install

# Restart dev server
npm run dev
```

### API connection fails
- Verify Flask backend is running: `Invoke-WebRequest -Uri http://localhost:5000/health`
- Check CORS is enabled in `Backend/app.py`
- Ensure both servers are on the same machine or update API URL

### Model loading error
- Verify model file exists: `d:/MINI project/ml-service/tamil_deepfake/models/best_model.pth`
- Check PyTorch installation: `python -c "import torch; print(torch.__version__)"`
- Ensure sufficient disk space

## 📦 Building for Production

### Frontend Build
```bash
cd Front
npm run build

# Output: dist/ folder with optimized assets
# Serve with any web server (Nginx, Apache, etc.)
```

### Backend Deployment
```bash
# Install production server
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Or Docker
docker build -t deepfake-api .
docker run -p 5000:5000 deepfake-api
```

## 📊 Usage Statistics

After deployment, you can track:
- Total predictions made
- Average processing time
- Model accuracy in production
- File format distribution
- Peak usage hours

## 🔐 Security

For production deployment:
1. Enable HTTPS/SSL
2. Implement rate limiting
3. Add input validation
4. Use environment variables for secrets
5. Deploy behind reverse proxy (Nginx)
6. Add authentication if needed
7. Monitor API usage

## 🚀 Enhancement Ideas

- [ ] Add visualization of model decision factors
- [ ] Support for multiple languages
- [ ] Batch processing for multiple files
- [ ] User accounts and history
- [ ] Export results as PDF
- [ ] Browser audio recording
- [ ] Real-time streaming analysis
- [ ] Model fine-tuning with user feedback

## 📚 Documentation

- **`SETUP_GUIDE.md`** - Detailed setup instructions
- **`TECHNICAL_DOCUMENTATION.md`** - Architecture and implementation
- **`START_ALL.bat`** - Quick start launcher (Windows)
- **`START_ALL.ps1`** - Quick start launcher (PowerShell)

## 🤝 Contributing

To contribute improvements:
1. Modify React components in `Front/src/App.jsx`
2. Update API endpoints in `Backend/app.py`
3. Retrain model with new data
4. Test thoroughly before deployment

## 📄 License

This project is for educational and research purposes.

## 👥 Credits

Built with ❤️ for Tamil speech analysis and deepfake detection.

---

## 📞 Quick Links

| Link | Purpose |
|------|---------|
| http://localhost:5173 | Frontend UI |
| http://localhost:5000 | API Server |
| http://localhost:5000/health | API Health Check |
| http://localhost:5000/api/info | API Documentation |

## 🎯 Next Steps

1. ✅ Start both servers (use `START_ALL.bat`)
2. ✅ Open http://localhost:5173
3. ✅ Upload a Tamil audio file
4. ✅ Click "Analyze"
5. ✅ View results!

---

**Happy Detecting! 🎉**

*For questions or issues, check the documentation files or test endpoints using Invoke-WebRequest in PowerShell.*
