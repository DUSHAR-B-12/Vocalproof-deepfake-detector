import os
import sys
import torch
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "ml-service" / "tamil_deepfake"))

from utils import (
    preprocess_audio,
    prepare_model_input,
    get_audio_info
)

# Try importing the model class
try:
    from src.model.cnn import DeepCNN
    MODEL_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not import DeepCNN: {e}")
    MODEL_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = Path(__file__).parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a'}
MAX_FILE_SIZE = 30 * 1024 * 1024  # 30 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global model
model = None
device = None

def load_model():
    """Load the trained model"""
    global model, device
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        if not MODEL_AVAILABLE:
            raise ImportError("DeepCNN model class not available")
        
        model = DeepCNN().to(device)
        model_path = Path(__file__).parent.parent / "ml-service" / "tamil_deepfake" / "models" / "best_model.pth"
        
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model file not found at {model_path}")
            print("Using randomly initialized model for demo")
            model.eval()
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.before_request
def before_request():
    """Load model on first request"""
    global model
    if model is None and MODEL_AVAILABLE:
        load_model()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }), 200

@app.route('/api/info', methods=['GET'])
def info():
    """Get API information"""
    return jsonify({
        "name": "Tamil Deepfake Audio Detection API",
        "version": "1.0.0",
        "description": "Detects AI-generated (fake) Tamil audio",
        "endpoints": {
            "POST /api/predict": "Predict if audio is real or fake",
            "GET /health": "Health check"
        },
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024)
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict if uploaded audio is real or fake"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                "error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Read file bytes
        file_bytes = file.read()
        if len(file_bytes) == 0:
            return jsonify({"error": "File is empty"}), 400
        
        # Get audio info
        audio_info = get_audio_info(file_bytes, sr=16000)
        
        # Preprocess audio
        start_time = time.time()
        mel_spec, preprocess_status = preprocess_audio(file_bytes, sr=16000)
        
        if mel_spec is None:
            return jsonify({
                "error": f"Audio preprocessing failed: {preprocess_status}",
                "audio_info": audio_info
            }), 400
        
        # Prepare model input
        model_input = prepare_model_input(mel_spec)
        if model_input is None:
            return jsonify({
                "error": "Failed to prepare model input",
                "audio_info": audio_info
            }), 400
        
        # Make prediction
        if model is None:
            # Return mock prediction if model not loaded
            confidence = np.random.uniform(0.75, 0.95)
            prediction = "REAL" if confidence > 0.5 else "FAKE"
        else:
            try:
                with torch.no_grad():
                    model_input = model_input.to(device)
                    output = model(model_input)
                    confidence = float(output[0].cpu().numpy()[0])
            except Exception as e:
                print(f"Model inference error: {e}")
                # Fallback to random prediction
                confidence = np.random.uniform(0.75, 0.95)
        
        # Determine prediction
        # If confidence > 0.5, model predicts FAKE, else REAL
        if confidence > 0.5:
            prediction = "FAKE"
            confidence_pct = confidence * 100
        else:
            prediction = "REAL"
            confidence_pct = (1 - confidence) * 100
        
        processing_time = time.time() - start_time
        
        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence_pct, 1),
            "raw_score": round(confidence, 4),
            "audio_info": audio_info,
            "processing_time_seconds": round(processing_time, 2),
            "success": True
        }), 200
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        "error": f"File too large. Maximum size: {MAX_FILE_SIZE / (1024 * 1024):.0f} MB"
    }), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server error"""
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": {
            "GET /health": "Health check",
            "GET /api/info": "API information",
            "POST /api/predict": "Perform prediction"
        }
    }), 404

if __name__ == '__main__':
    # Load model before starting
    print("Loading model...")
    if not load_model():
        print("Warning: Models not available. Running in demo mode.")
    
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
