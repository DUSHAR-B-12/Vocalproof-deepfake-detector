from pathlib import Path
from src.preprocessing.preprocess import run_preprocessing

print('Starting preprocessing runner')
DATA_ROOT = Path(r"D:/MINI project/ml-service/data/Audio_Dataset")
if not DATA_ROOT.exists():
    DATA_ROOT = Path(r"D:/MINI project/ml-service/data/audio_dataset")
    if not DATA_ROOT.exists():
        raise SystemExit('Dataset not found at expected locations')

OUT_ROOT = Path('.') / 'data'
CFG = Path('config/config.yaml')
print('Using dataset:', DATA_ROOT)
print('Output root:', OUT_ROOT)
run_preprocessing(DATA_ROOT, OUT_ROOT, CFG)
print('Preprocessing complete')
