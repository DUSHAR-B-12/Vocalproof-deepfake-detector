import streamlit as st
from pathlib import Path
import tempfile
import librosa
import librosa.display
import numpy as np
import soundfile as sf
from yaml import safe_load
import torch

from src.model.cnn import DeepCNN


def load_config(path: Path) -> dict:
    with open(path, 'r') as f:
        return safe_load(f)


@st.cache_resource
def load_model(model_path: Path, device: str = 'cpu'):
    dev = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model = DeepCNN().to(dev)
    model.load_state_dict(torch.load(model_path, map_location=dev))
    model.eval()
    return model, dev


def preprocess_audio(y, sr, duration=3.0):
    target = int(sr * duration)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def main():
    st.title('Tamil Deepfake Audio Detection')
    app_dir = Path(__file__).parent
    cfg = load_config(app_dir / 'config/config.yaml')
    model_path = app_dir / 'models/best_model.pth'
    if model_path.exists():
        model, device = load_model(model_path, device='cpu')
    else:
        model = None
        device = 'cpu'

    uploaded = st.file_uploader('Upload audio', type=['wav', 'mp3', 'flac'])
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        y, sr = librosa.load(tmp_path, sr=cfg.get('sr', 16000))
        mel = preprocess_audio(y, sr, duration=cfg.get('duration', 3.0))
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3))
        librosa.display.specshow(mel, sr=sr, x_axis='time', y_axis='mel', fmax=8000, ax=ax)
        ax.set_title('Mel Spectrogram')
        st.pyplot(fig)
        if model is None:
            st.warning('Model not found. Run pipeline to train a model.')
        else:
            arr = np.expand_dims(mel, 0)
            arr = np.expand_dims(arr, 0)
            tensor = torch.tensor(arr, dtype=torch.float32).to(device)
            with torch.no_grad():
                out = model(tensor)
                prob = float(out.cpu().numpy().squeeze())
                label = 'REAL' if prob >= 0.5 else 'FAKE'
                color = 'green' if label == 'REAL' else 'red'
                st.markdown(f"<h2 style='color:{color}'>{label} ({prob*100:.2f}%)</h2>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
