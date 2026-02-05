from pathlib import Path
from src.features.extract_features import extract_and_save

print('Starting feature extraction')
PROCESSED_ROOT = Path('data/processed')
FEATURES_ROOT = Path('data/features')
CFG = Path('config/config.yaml')

extract_and_save(PROCESSED_ROOT, FEATURES_ROOT, CFG)
print('Feature extraction complete')
