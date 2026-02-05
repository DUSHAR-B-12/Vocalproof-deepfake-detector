from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(log_dir: Path, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.title('Training curves placeholder')
    plt.savefig(out_path)


def show_spectrogram(npy_path: Path, out_path: Path):
    arr = np.load(npy_path)
    plt.figure(figsize=(6, 4))
    plt.imshow(arr, origin='lower', aspect='auto')
    plt.colorbar()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
