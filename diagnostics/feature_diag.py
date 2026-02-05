from pathlib import Path
import traceback
import numpy as np
import librosa
from yaml import safe_load

CFG = Path('config/config.yaml')
PROCESSED_ROOT = Path('data/processed')
FEATURES_ROOT = Path('data/features')

with open(CFG,'r') as f:
    cfg = safe_load(f)

sr = cfg.get('sr',16000)
n_mels = cfg.get('n_mels',128)
n_mfcc = cfg.get('n_mfcc',40)

for label in ['real','fake']:
    in_dir = PROCESSED_ROOT / label
    mel_out = FEATURES_ROOT / 'mel_spectrograms' / label
    mfcc_out = FEATURES_ROOT / 'mfcc' / label
    mel_out.mkdir(parents=True, exist_ok=True)
    mfcc_out.mkdir(parents=True, exist_ok=True)
    files = list(in_dir.glob('*.wav'))[:5]
    print(f'label={label}, found {len(files)} files (showing up to 5)')
    for f in files:
        try:
            y,_ = librosa.load(f, sr=sr)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=n_mfcc)
            mel_path = mel_out / (f.stem + '.npy')
            mfcc_path = mfcc_out / (f.stem + '.npy')
            print('Saving', mel_path)
            np.save(mel_path, mel_db)
            np.save(mfcc_path, mfcc)
        except Exception as e:
            print('Error processing', f)
            traceback.print_exc()

print('Done. Directory listing for features:')
for p in FEATURES_ROOT.rglob('*'):
    print(p)