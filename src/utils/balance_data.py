import random
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
from yaml import safe_load


def load_config(path: Path) -> dict:
    with open(path, 'r') as f:
        return safe_load(f)


def add_noise(y: np.ndarray, snr_db: float) -> np.ndarray:
    rms = np.sqrt(np.mean(y**2))
    snr = 10**(snr_db / 20.0)
    noise_rms = rms / snr
    noise = np.random.normal(0, noise_rms, y.shape)
    return y + noise


def pitch_shift(y: np.ndarray, sr: int, steps: int) -> np.ndarray:
    return librosa.effects.pitch_shift(y, sr, steps)


def time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    try:
        return librosa.effects.time_stretch(y, rate)
    except Exception:
        return y


def check_and_balance(processed_root: Path, config_path: Path):
    cfg = load_config(config_path)
    aug_cfg = cfg.get('augmentations', {})
    real_files = list((processed_root / 'real').glob('*.wav'))
    fake_files = list((processed_root / 'fake').glob('*.wav'))
    n_real = len(real_files)
    n_fake = len(fake_files)
    print('Counts -> real:', n_real, 'fake:', n_fake)
    if n_real == 0 or n_fake == 0:
        print('One of the classes is empty; skipping augmentation.')
        return
    target = max(n_real, n_fake)
    minority = 'real' if n_real < n_fake else 'fake'
    minority_files = real_files if minority == 'real' else fake_files
    sr = cfg.get('sr', 16000)
    i = 0
    while len(minority_files) < target:
        src = random.choice(minority_files)
        y, _ = librosa.load(src, sr=sr)
        op = random.choice(['pitch', 'stretch', 'noise'])
        if op == 'pitch':
            steps = random.choice(aug_cfg.get('pitch_steps', [1]))
            y2 = pitch_shift(y, sr, steps)
        elif op == 'stretch':
            rate = random.choice(aug_cfg.get('time_stretch', [1.0]))
            y2 = time_stretch(y, rate)
            if len(y2) < int(cfg.get('sr', 16000) * cfg.get('duration', 3.0)):
                y2 = np.pad(y2, (0, 1))
        else:
            snr = random.choice(aug_cfg.get('noise_snr_db', [20]))
            y2 = add_noise(y, snr)
        outp = processed_root / minority / f"aug_{i}_{src.name}"
        sf.write(str(outp) + '.wav', y2, sr)
        minority_files.append(outp)
        i += 1
    print('Balanced classes by augmenting', i, 'files.')
