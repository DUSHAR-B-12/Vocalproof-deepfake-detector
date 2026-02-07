"""
Debug script to verify model output direction and label mapping
"""
import sys
import torch
import numpy as np
from pathlib import Path

# Setup paths
BACKEND_DIR = Path(__file__).parent
ML_SERVICE_DIR = BACKEND_DIR.parent / "ml-service" / "tamil_deepfake"
MODEL_PATH = ML_SERVICE_DIR / "models" / "best_model.pth"

sys.path.insert(0, str(ML_SERVICE_DIR))
sys.path.insert(0, str(BACKEND_DIR))

print("\n" + "="*70)
print("🔍 MODEL OUTPUT DIRECTION DEBUG")
print("="*70)

# Check CSV labels
print("\n1️⃣ CHECKING LABEL MAPPING FROM CSV:")
import pandas as pd
train_csv = ML_SERVICE_DIR / "data" / "splits" / "train.csv"
df = pd.read_csv(train_csv)

print(f"\nSample FAKE files (should all have label 0):")
fake_samples = df[df['file'].str.contains('fake', case=False)].head(3)
print(fake_samples[['file', 'label']])

print(f"\nSample REAL files (should all have label 1):")
real_samples = df[df['file'].str.contains('real', case=False)].head(3)
print(real_samples[['file', 'label']])

# Load model
print("\n2️⃣ LOADING MODEL:")
try:
    from src.model.cnn import DeepCNN
    device = torch.device('cpu')
    model = DeepCNN().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# Test with dummy inputs
print("\n3️⃣ TESTING MODEL WITH DUMMY INPUTS:")

# Create dummy input
dummy_input = torch.randn(1, 1, 128, 100).to(device)

with torch.no_grad():
    output = model(dummy_input)
    print(f"\nDummy input shape: {dummy_input.shape}")
    print(f"Model output shape: {output.shape}")
    print(f"Model output: {output}")
    print(f"Output value (as float): {float(output[0].cpu().numpy())}")
    
# Load actual feature files to test
print("\n4️⃣ TESTING WITH ACTUAL FEATURES:")

# Load a FAKE sample
fake_file = ML_SERVICE_DIR / "data" / "features" / "mel_spectrograms" / "fake" / "ai_0_0.npy"
real_file = ML_SERVICE_DIR / "data" / "features" / "mel_spectrograms" / "real" / list((ML_SERVICE_DIR / "data" / "features" / "mel_spectrograms" / "real").glob("*.npy"))[0].name

print(f"\nFAKE file: {fake_file.name}")
print(f"REAL file: {real_file}")

try:
    # Test FAKE
    if fake_file.exists():
        fake_mel = np.load(fake_file)
        if fake_mel.ndim == 2:
            fake_mel = np.expand_dims(fake_mel, 0)
        fake_tensor = torch.tensor(fake_mel, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            fake_output = model(fake_tensor)
            fake_confidence = float(fake_output[0].cpu().numpy())
        
        print(f"  Mel shape: {fake_mel.shape}")
        print(f"  Model output: {fake_confidence:.4f}")
        print(f"  Expected: should be close to 0 (label 0 = FAKE)")
        print(f"  Current interpretation: {'REAL ❌' if fake_confidence >= 0.5 else 'FAKE ✅'}")
    
    # Test REAL
    real_path = ML_SERVICE_DIR / "data" / "features" / "mel_spectrograms" / "real" / real_file
    if real_path.exists():
        real_mel = np.load(real_path)
        if real_mel.ndim == 2:
            real_mel = np.expand_dims(real_mel, 0)
        real_tensor = torch.tensor(real_mel, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            real_output = model(real_tensor)
            real_confidence = float(real_output[0].cpu().numpy())
        
        print(f"\n  Mel shape: {real_mel.shape}")
        print(f"  Model output: {real_confidence:.4f}")
        print(f"  Expected: should be close to 1 (label 1 = REAL)")
        print(f"  Current interpretation: {'REAL ✅' if real_confidence >= 0.5 else 'FAKE ❌'}")

except Exception as e:
    print(f"❌ Error loading/testing actual files: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("💡 INTERPRETATION:")
print("="*70)
print("""
If FAKE file output is close to 0 and REAL file output is close to 1:
  → Current code is CORRECT (confidence >= 0.5 → REAL)

If FAKE file output is close to 1 and REAL file output is close to 0:
  → Current code is WRONG (need to swap the logic)
""")
print("="*70 + "\n")
