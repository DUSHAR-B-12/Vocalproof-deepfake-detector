from pathlib import Path
from src.utils.create_splits import create_stratified_splits

print('Creating stratified splits')
FEATURES_ROOT = Path('data/features')
SPLITS_ROOT = Path('data/splits')
CFG = Path('config/config.yaml')
create_stratified_splits(FEATURES_ROOT, SPLITS_ROOT, CFG)
print('Splits created')
