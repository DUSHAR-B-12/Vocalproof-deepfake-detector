from pathlib import Path
from typing import List
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
from tqdm import tqdm
from yaml import safe_load


def load_config(path: Path) -> dict:
    with open(path, 'r') as f:
        return safe_load(f)


def is_supported(file: Path, exts: List[str]) -> bool:
    return file.suffix.lower() in exts


def denoise(y: np.ndarray, sr: int) -> np.ndarray:
    try:
        return nr.reduce_noise(y=y, sr=sr)
    except Exception:
        return y


def trim_silence(y: np.ndarray, sr: int) -> np.ndarray:
    yt, _ = librosa.effects.trim(y)
    return yt


def normalize(y: np.ndarray) -> np.ndarray:
    maxv = np.max(np.abs(y))
    if maxv > 0:
        return y / maxv
    return y


def segment_audio(y: np.ndarray, sr: int, duration: float) -> List[np.ndarray]:
    seg_len = int(sr * duration)
    if len(y) <= seg_len:
        pad = seg_len - len(y)
        return [np.pad(y, (0, pad))]
    segments = []
    step = seg_len
    for start in range(0, len(y), step):
        seg = y[start:start + seg_len]
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)))
        segments.append(seg)
    return segments


def process_file(infile: Path, outdir: Path, sr: int, duration: float, exts: List[str]):
    try:
        if not is_supported(infile, exts):
            return 0
        y, _ = librosa.load(infile, sr=sr, mono=True)
        y = denoise(y, sr)
        y = trim_silence(y, sr)
        y = normalize(y)
        segs = segment_audio(y, sr, duration)
        saved = 0
        for i, seg in enumerate(segs):
            outpath = outdir / f"{infile.stem}_{i}.wav"
            sf.write(outpath, seg, sr)
            saved += 1
        return saved
    except Exception:
        return 0


def run_preprocessing(root_dataset: Path, out_root: Path, config_path: Path):
    cfg = load_config(config_path)
    sr = cfg.get('sr', 16000)
    duration = cfg.get('duration', 3.0)
    exts = cfg.get('supported_extensions', ['.wav'])

    for label in ['real', 'fake']:
        in_dir = root_dataset / label
        out_dir = out_root / 'processed' / label
        out_dir.mkdir(parents=True, exist_ok=True)
        if not in_dir.exists():
            print('Input directory missing:', in_dir)
            continue
        files = list(in_dir.rglob('*'))
        total = 0
        for f in tqdm(files, desc=f'Processing {label}'):
            if f.is_file():
                total += process_file(f, out_dir, sr, duration, exts)
        print(f'Processed {total} segments for', label)
