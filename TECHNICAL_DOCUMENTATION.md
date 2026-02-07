# Technical Documentation - Tamil Deepfake Detection System

## Backend Architecture

### Flask Application Structure

#### `Backend/app.py`
Main Flask application with the following key functions:

**Global State:**
- `model`: Loaded CNN model instance
- `device`: PyTorch device (CPU/GPU)

**Key Functions:**

1. **`load_model()`**
   - Initializes PyTorch DeepCNN model
   - Loads trained weights from `.pth` file
   - Sets model to evaluation mode
   - Returns success/failure status

2. **Routes:**

   **`@app.route('/health', methods=['GET'])`**
   - Returns API health status
   - Indicates if model is loaded
   - Shows compute device being used

   **`@app.route('/api/info', methods=['GET'])`**
   - Returns API documentation
   - Lists all available endpoints
   - Shows supported file formats and size limits

   **`@app.route('/api/predict', methods=['POST'])`**
   - Main prediction endpoint
   - Validates uploaded file
   - Preprocesses audio
   - Runs model inference
   - Returns prediction with confidence score

3. **Error Handlers:**
   - 413: File too large
   - 404: Endpoint not found
   - 500: Internal server error

### Audio Processing Pipeline

#### `Backend/utils.py`
Utility functions for audio preprocessing and feature extraction:

**1. Audio Loading (`load_audio`)**
```python
def load_audio(file_path_or_bytes, sr=16000):
    # Handles both file paths and byte streams
    # Resamples to 16kHz standard
    # Returns audio array and sample rate
```

**2. Audio Preprocessing (`preprocess_audio`)**
Complete pipeline:
```
Raw Audio → Trim Silence → Normalize → Extract Mel Spectrogram → Return Features
```

**3. Feature Extraction**
- **Mel Spectrogram:** 128 frequency bins, standard audio feature
- **MFCC:** Mel-frequency cepstral coefficients (alternative feature)

**4. Model Input Preparation (`prepare_model_input`)**
```
Mel Spec (2D) → Add Channel Dim (3D) → Normalize → Tensor → Add Batch Dim (4D)
Input shape: (1, 1, 128, Freq_Bins)
```

### Request-Response Flow

```
1. Client uploads audio file
   ↓
2. Flask receives multipart form data
   ↓
3. File validation (size, extension)
   ↓
4. Audio loading from bytes
   ↓
5. Silence trimming
   ↓
6. Audio normalization
   ↓
7. Mel spectrogram extraction
   ↓
8. Input preparation for model
   ↓
9. PyTorch inference (no_grad context)
   ↓
10. Score interpretation
    - Score > 0.5 → FAKE
    - Score ≤ 0.5 → REAL
   ↓
11. Confidence calculation
    - FAKE: score × 100
    - REAL: (1 - score) × 100
   ↓
12. JSON response with results
```

## Frontend Architecture

### React Component Hierarchy

```
App (Main Container)
├── Header
│   ├── Logo & Title
│   └── Navigation
├── Hero
│   ├── Title (Gradient Animated)
│   ├── Description
│   └── CTA Buttons
├── Main Content Grid (2-col layout)
│   ├── Left Column (2/3 width)
│   │   └── Upload Section
│   │       ├── FileUploader
│   │       ├── AudioPlayer
│   │       ├── Error Display
│   │       ├── Analyze Button
│   │       ├── ResultCard
│   │       ├── SpectrogramChart
│   │       ├── WaveformDisplay
│   │       └── StatsCards
│   └── Right Sidebar (1/3 width)
│       ├── ModelInfoCard
│       └── Sample Predictions
├── About Section
│   ├── How It Works
│   └── Technology Stack
└── Footer
```

### State Management

**Main `App` Component State:**
```javascript
const [file, setFile]                   // Currently uploaded file
const [processing, setProcessing]       // API call in progress
const [result, setResult]               // "REAL" or "FAKE"
const [confidence, setConfidence]       // Confidence percentage (0-100)
const [error, setError]                 // Error message if any
const [audioInfo, setAudioInfo]         // Metadata from API
const [processingTime, setProcessingTime] // API response time
```

### API Integration

**API Base URL Configuration:**
```javascript
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000'
```

Environment variables can be set via `.env` file:
```
REACT_APP_API_URL=http://localhost:5000
```

**Main Prediction Flow:**
```javascript
async function analyze() {
    1. Validate file exists
    2. Create FormData with file
    3. POST to /api/predict
    4. Parse JSON response
    5. Update state with results
    6. Handle errors gracefully
}
```

### Styling System

**Tailwind CSS Usage:**
- Core utilities only (no custom CSS classes)
- Dark mode configuration in `tailwind.config.cjs`
- Color palette mapped to Tailwind extends

**Custom CSS Classes (in `src/index.css`):**
- `.gradient-anim`: Animated gradient text effect
- `.glass`: Glassmorphism effect with blur and translucency
- `.glow`: Box shadow glow effect

**Animation Examples:**
```css
/* Animated gradient text */
.gradient-anim {
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #ec4899, #60a5fa);
    background-size: 300% 300%;
    animation: gradientShift 6s ease infinite;
}

/* Glassmorphism */
.glass {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.06);
}
```

### Component Details

**FileUploader Component:**
- Drag-drop support with hover effects
- File input button fallback
- Validates file extensions
- Updates parent state on selection

**CircularConfidence Component:**
- SVG-based circular progress indicator
- Animated stroke dash animation
- Color changes based on prediction (green/red)
- Displays percentage in center

**ResultCard Component:**
- Badge showing REAL/FAKE with emoji
- Circular confidence indicator
- Processing time display
- Detailed explanation of result

**StatsCards Component:**
- Display audio metadata from API response
- Duration, sample rate, file size
- Falls back to --  if data unavailable

**ModelInfoCard Component:**
- Animated counter for accuracy metric
- Grid of performance metrics
- Training data information
- Uses useEffect for animation

## Data Flow Diagrams

### Upload & Prediction Flow

```
┌──────────────────┐
│ User uploads file│
└────────┬─────────┘
         │
         ↓
┌──────────────────────────────┐
│ FileUploader validates      │
│ - Extension check          │
│ - Basic file validation    │
└────────┬─────────────────────┘
         │
         ↓
┌──────────────────────────────┐
│ User clicks "Analyze"      │
│ Create FormData            │
│ POST /api/predict          │
└────────┬─────────────────────┘
         │
         ↓
    ┌─────────────────┐
    │ Flask Backend   │
    └────────┬────────┘
             │
             ↓
    ┌──────────────────────────────┐
    │ Validate file                │
    │ Load audio from bytes        │
    │ Preprocess (trim, normalize) │
    │ Extract mel spectrogram      │
    │ Prepare model input          │
    └────────┬─────────────────────┘
             │
             ↓
    ┌──────────────────────────────┐
    │ PyTorch Model Inference      │
    │ CNN forward pass             │
    │ Get confidence score         │
    └────────┬─────────────────────┘
             │
             ↓
    ┌──────────────────────────────┐
    │ Interpret results            │
    │ Score > 0.5 → FAKE           │
    │ Score ≤ 0.5 → REAL           │
    │ Build JSON response          │
    └────────┬─────────────────────┘
             │
         ↓
┌──────────────────────────────┐
│ React receives response     │
│ Update state                │
│ Render result card          │
│ Display visualizations      │
│ Show audio stats            │
└──────────────────────────────┘
```

## Performance Considerations

### Frontend
- **Bundle Size:** ~150KB (gzipped with React + dependencies)
- **Bundle Time:** < 2s on typical connection
- **Development Server:** Vite provides hot module reload
- **Build Output:** Optimized static assets in `dist/` folder

### Backend
- **Model Size:** ~50MB (PyTorch .pth file)
- **Memory Usage:** ~500MB-1GB (model + processing)
- **Processing Time:** ~500ms-2s per audio file
- **Concurrent Requests:** Flask handles multiple via threading

### Optimization Tips
1. **Frontend:**
   - Lazy load visualizations
   - Memoize components with `React.memo()`
   - Use production build: `npm run build`

2. **Backend:**
   - Cache model after initial load
   - Use batch processing for multiple files
   - Deploy with Gunicorn for production
   - Use GPU if available (update device selection)

## Deployment Considerations

### Production Frontend
```bash
# Build optimized frontend
cd Front
npm run build

# Serve with a web server (e.g., Nginx, Express)
# Point to dist/ folder
```

### Production Backend
```bash
# Install production WSGI server
pip install gunicorn

# Run with Gunicorn (4 workers as example)
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Or with uWSGI
uwsgi --http :5000 --wsgi-file app.py --callable app --processes 4 --threads 2
```

### Docker Setup (Optional)

**Backend Dockerfile:**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

**Frontend Dockerfile:**
```dockerfile
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json .
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## Environment Variables

### Frontend (`.env`)
```
VITE_API_URL=http://localhost:5000
# or for production:
# VITE_API_URL=https://api.example.com
```

Used as: `import.meta.env.VITE_API_URL`

### Backend (`.env` or environment)
```
FLASK_ENV=development
FLASK_DEBUG=False
UPLOAD_FOLDER=./uploads
MODEL_PATH=../ml-service/tamil_deepfake/models/best_model.pth
DEVICE=auto  # or 'cpu', 'cuda'
```

## Testing

### Backend Testing
```python
# Test health endpoint
curl http://localhost:5000/health

# Test with sample audio file
form_data = {
    'file': open('sample.wav', 'rb')
}
requests.post('http://localhost:5000/api/predict', files=form_data)
```

### Frontend Testing
```javascript
// Test API connection in browser console
fetch('http://localhost:5000/health')
    .then(r => r.json())
    .then(d => console.log(d))
```

## Monitoring & Debugging

### Backend Logging
Enable debug logging in `app.py`:
```python
app.run(debug=True)  # Development only
```

### Frontend Debugging
- Use React DevTools browser extension
- Open browser console (F12) for network/error logs
- Check network tab for API requests
- Use `console.log()` for debugging

### Common Issues & Solutions

1. **CORS Errors**
   - Ensure Flask-CORS is installed
   - Verify CORS headers are sent: `https://localhost:5173` should work

2. **Model Loading Failures**
   - Check file path is correct
   - Verify `.pth` file exists and is readable
   - Ensure PyTorch is properly installed

3. **Audio Processing Errors**
   - Verify librosa is installed: `pip install librosa`
   - Check audio file format is supported
   - Ensure file is not corrupted

4. **API Connection Issues**
   - Verify Flask server is running
   - Check port 5000 is not in use
   - Verify CORS is enabled in Flask

---

For more information, see `SETUP_GUIDE.md`
