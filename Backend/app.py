import os
import sys
import torch
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import time
import traceback

# Get absolute paths
BACKEND_DIR = Path(__file__).parent
ML_SERVICE_DIR = BACKEND_DIR.parent / "ml-service" / "tamil_deepfake"
MODEL_PATH = ML_SERVICE_DIR / "models" / "best_model.pth"

# Add ML service to path for imports
sys.path.insert(0, str(ML_SERVICE_DIR))
sys.path.insert(0, str(BACKEND_DIR))

print(f"\n{'='*60}")
print(f" TAMIL DEEPFAKE DETECTION API - STARTUP")
print(f"{'='*60}")
print(f"Backend directory: {BACKEND_DIR}")
print(f"ML Service directory: {ML_SERVICE_DIR}")
print(f"Model path: {MODEL_PATH}")
print(f"Model exists: {MODEL_PATH.exists()}")

from utils import (
    preprocess_audio,
    prepare_model_input,
    get_audio_info
)

# Try importing the model class
MODEL_AVAILABLE = False
DeepCNN = None
try:
    from src.model.cnn import DeepCNN
    MODEL_AVAILABLE = True
    print("[OK] DeepCNN imported successfully from src.model.cnn")
except Exception as e:
    print(f"[WARNING] Could not import DeepCNN from src.model.cnn: {e}")
    print(f"   Trying alternative import...")
    try:
        # Try alternative import path
        import importlib.util
        cnn_path = ML_SERVICE_DIR / "src" / "model" / "cnn.py"
        spec = importlib.util.spec_from_file_location("cnn", cnn_path)
        if spec and spec.loader:
            cnn_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cnn_module)
            DeepCNN = cnn_module.DeepCNN
            MODEL_AVAILABLE = True
            print(f"[OK] DeepCNN imported via alternative method from {cnn_path}")
        else:
            print(f"[ERROR] Could not load module spec from {cnn_path}")
    except Exception as e2:
        print(f"[ERROR] Alternative import also failed: {e2}")

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
model_loaded = False

def load_model():
    """Load the trained model"""
    global model, device, model_loaded
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n[INFO] Using device: {device.type.upper()}")
        
        if not MODEL_AVAILABLE or DeepCNN is None:
            print("[ERROR] DeepCNN class not available - cannot load model")
            model_loaded = False
            return False
        
        print("[INFO] Creating DeepCNN model architecture...")
        model = DeepCNN().to(device)
        
        if not MODEL_PATH.exists():
            print(f"[ERROR] Model file not found at {MODEL_PATH}")
            print("   Cannot proceed without trained model weights")
            model = None
            model_loaded = False
            return False
        
        print(f"[INFO] Loading trained weights from {MODEL_PATH}...")
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f"[OK] MODEL READY - Successfully loaded and ready for inference")
        model_loaded = True
        print(f"{'='*60}\n")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        print(traceback.format_exc())
        model = None
        model_loaded = False
        print(f"{'='*60}\n")
        return False


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.before_request
def before_request():
    """Load model on first request"""
    global model, model_loaded
    if not model_loaded:
        print("\n[INFO] First request detected - Loading model...")
        load_model()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded,
        "device": str(device.type) if device else "unknown"
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
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
        "model_status": "READY" if model_loaded else "NOT LOADED"
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict if uploaded audio is real or fake"""
    try:
        # Check if model is loaded
        if not model_loaded or model is None:
            return jsonify({
                "error": "Model not loaded. Please restart the server.",
                "success": False
            }), 503
        
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
        
        print(f"\n[INFO] Processing file: {file.filename} ({len(file_bytes)} bytes)")
        
        # Get audio info
        audio_info = get_audio_info(file_bytes, sr=16000)
        print(f"   Audio info: {audio_info}")
        
        # Preprocess audio
        start_time = time.time()
        mel_spec, preprocess_status = preprocess_audio(file_bytes, sr=16000)
        
        if mel_spec is None:
            return jsonify({
                "error": f"Audio preprocessing failed: {preprocess_status}",
                "audio_info": audio_info,
                "success": False
            }), 400
        
        print(f"   Mel spectrogram shape: {mel_spec.shape}")
        
        # Prepare model input
        model_input = prepare_model_input(mel_spec)
        if model_input is None:
            return jsonify({
                "error": "Failed to prepare model input",
                "audio_info": audio_info,
                "success": False
            }), 400
        
        print(f"   Model input shape: {model_input.shape}")
        
        # Make prediction with actual model
        try:
            print(f"   Running model inference...")
            with torch.no_grad():
                model_input_device = model_input.to(device)
                output = model(model_input_device)
                confidence = float(output[0].cpu().numpy())
            
            print(f"   Model output (raw): {confidence:.4f}")
            
            # Interpret the confidence score
            # Model was trained with:
            # Label 0 = FAKE (AI-generated, ai_* files)
            # Label 1 = REAL (Human speech, human_* files)
            # Sigmoid output: 0-1 range
            # Score closer to 0 = FAKE, Score closer to 1 = REAL
            
            if confidence >= 0.5:
                prediction = "REAL"
                confidence_pct = confidence * 100
            else:
                prediction = "FAKE"
                confidence_pct = (1 - confidence) * 100
            
            processing_time = time.time() - start_time
            
            print(f"   [OK] Prediction: {prediction} ({confidence_pct:.1f}%)")
            print(f"   Processing time: {processing_time:.2f}s\n")
            
            return jsonify({
                "prediction": prediction,
                "confidence": round(confidence_pct, 1),
                "raw_score": round(confidence, 4),
                "audio_info": audio_info,
                "processing_time_seconds": round(processing_time, 2),
                "success": True
            }), 200
            
        except Exception as inference_error:
            print(f"[ERROR] Model inference error: {inference_error}")
            print(traceback.format_exc())
            return jsonify({
                "error": f"Model inference failed: {str(inference_error)}",
                "audio_info": audio_info,
                "success": False
            }), 500
        
    except Exception as e:
        print(f"[ERROR] Prediction endpoint error: {e}")
        print(traceback.format_exc())
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
    print("\n[INFO] Loading model at startup...")
    if load_model():
        print("[OK] Backend ready to start\n")
    else:
        print("[WARNING] Model could not be loaded. API will return errors.\n")
    
    print("[INFO] Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

