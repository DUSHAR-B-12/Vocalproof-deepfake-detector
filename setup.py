import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

FOLDERS = [
    PROJECT_ROOT / 'data' / 'processed' / 'real',
    PROJECT_ROOT / 'data' / 'processed' / 'fake',
    PROJECT_ROOT / 'data' / 'features' / 'mel_spectrograms',
    PROJECT_ROOT / 'data' / 'features' / 'mfcc',
    PROJECT_ROOT / 'data' / 'splits',
    PROJECT_ROOT / 'models',
    PROJECT_ROOT / 'logs',
    PROJECT_ROOT / 'logs' / 'tensorboard',
    PROJECT_ROOT / 'visualizations',
]

DATASET_PATHS = [
    Path('ml-service/data/Audio_Dataset'),
    Path('ml-service/data/audio_dataset'),
    Path('ML-services/data/audio_dataset'),
]


def create_folders():
    for p in FOLDERS:
        p.mkdir(parents=True, exist_ok=True)
    print('Created project folders under', PROJECT_ROOT)


def verify_dataset():
    found = False
    for p in DATASET_PATHS:
        if p.exists():
            print('Found dataset root at', p)
            found = True
            break
    if not found:
        print('Warning: dataset path not found in expected locations:')
        for p in DATASET_PATHS:
            print(' -', p)


def install_requirements():
    req = PROJECT_ROOT / 'requirements.txt'
    if not req.exists():
        print('requirements.txt not found at', req)
        return
    print('Installing requirements from', req)
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', str(req)])
    except subprocess.CalledProcessError as e:
        print('pip install failed:', e)


def main():
    create_folders()
    verify_dataset()
    print('To install Python dependencies, run: python setup.py --install')


if __name__ == '__main__':
    if '--install' in sys.argv:
        install_requirements()
    else:
        main()
