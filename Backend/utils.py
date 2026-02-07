import librosa
import numpy as np
import torch
from pathlib import Path
import io

def load_audio(file_path_or_bytes, sr=16000):
    """Load audio from file path or bytes"""
    try:
        if isinstance(file_path_or_bytes, bytes):
            # Load from bytes (uploaded file)
            y, _ = librosa.load(io.BytesIO(file_path_or_bytes), sr=sr)
        else:
            # Load from file path
            y, _ = librosa.load(str(file_path_or_bytes), sr=sr)
        return y, sr
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None, None

def trim_silence(y, sr=16000, threshold_db=-40):
    """Remove silence from audio"""
    try:
        # Trim leading/trailing silence
        yt, _ = librosa.effects.trim(y, top_db=-threshold_db)
        return yt
    except Exception as e:
        print(f"Error trimming silence: {e}")
        return y

def normalize_audio(y):
    """Normalize audio to [-1, 1] range"""
    try:
        maxv = np.max(np.abs(y))
        if maxv > 0:
            return y / maxv
        return y
    except Exception as e:
        print(f"Error normalizing audio: {e}")
        return y

def extract_mel_spectrogram(y, sr=16000, n_mels=128):
    """Extract mel spectrogram features"""
    try:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db
    except Exception as e:
        print(f"Error extracting mel spectrogram: {e}")
        return None

def extract_mfcc(y, sr=16000, n_mfcc=40):
    """Extract MFCC features"""
    try:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=n_mfcc)
        return mfcc
    except Exception as e:
        print(f"Error extracting MFCC: {e}")
        return None

def preprocess_audio(file_bytes, sr=16000):
    """Complete preprocessing pipeline"""
    try:
        # Load audio
        y, sr = load_audio(file_bytes, sr=sr)
        if y is None:
            return None, "Failed to load audio file"
        
        # Trim silence
        y = trim_silence(y, sr=sr)
        
        # Normalize
        y = normalize_audio(y)
        
        # Extract features
        mel = extract_mel_spectrogram(y, sr=sr, n_mels=128)
        if mel is None:
            return None, "Failed to extract mel spectrogram"
        
        return mel, "success"
    except Exception as e:
        return None, str(e)

def prepare_model_input(mel_spec):
    """Prepare feature tensor for model inference"""
    try:
        # Ensure proper shape (C, H, W) for CNN
        if mel_spec.ndim == 2:
            mel_spec = np.expand_dims(mel_spec, axis=0)  # Add channel dim
        
        # Normalize to [0, 1]
        mel_min = mel_spec.min()
        mel_max = mel_spec.max()
        if mel_max > mel_min:
            mel_spec = (mel_spec - mel_min) / (mel_max - mel_min)
        
        # Convert to tensor
        tensor = torch.from_numpy(mel_spec).float()
        
        # Add batch dimension if needed
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)  # (1, C, H, W)
        
        return tensor
    except Exception as e:
        print(f"Error preparing model input: {e}")
        return None

def get_audio_info(file_bytes, sr=16000):
    """Get audio file information"""
    try:
        y, sr_actual = load_audio(file_bytes, sr=sr)
        if y is None:
            return None
        
        duration = librosa.get_duration(y=y, sr=sr_actual)
        file_size = len(file_bytes) / 1024  # KB
        
        return {
            "duration": round(duration, 2),
            "sample_rate": sr_actual,
            "file_size": round(file_size, 2),
            "samples": len(y)
        }
    except Exception as e:
        print(f"Error getting audio info: {e}")
        return None
