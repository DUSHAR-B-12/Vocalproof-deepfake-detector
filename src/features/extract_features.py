from pathlib import Path
import numpy as np
import librosa
from tqdm import tqdm
from yaml import safe_load


def load_config(path: Path) -> dict:
    with open(path, 'r') as f:
        return safe_load(f)


def extract_and_save(processed_root: Path, features_root: Path, config_path: Path):
    cfg = load_config(config_path)
    sr = cfg.get('sr', 16000)
    n_mels = cfg.get('n_mels', 128)
    n_mfcc = cfg.get('n_mfcc', 40)
    for label in ['real', 'fake']:
        in_dir = processed_root / label
        mel_out = features_root / 'mel_spectrograms' / label
        mfcc_out = features_root / 'mfcc' / label
        mel_out.mkdir(parents=True, exist_ok=True)
        mfcc_out.mkdir(parents=True, exist_ok=True)
        files = list(in_dir.glob('*.wav'))
        for f in tqdm(files, desc=f'Extracting {label}'):
            try:
                y, _ = librosa.load(f, sr=sr)
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=n_mfcc)
                np.save(mel_out / (f.stem + '.npy'), mel_db)
                np.save(mfcc_out / (f.stem + '.npy'), mfcc)
            except Exception:
                continue
