from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from yaml import safe_load


def load_config(path: Path) -> dict:
    with open(path, 'r') as f:
        return safe_load(f)


def create_stratified_splits(features_root: Path, splits_root: Path, config_path: Path):
    cfg = load_config(config_path)
    train_ratio = cfg.get('train_split', 0.7)
    val_ratio = cfg.get('val_split', 0.15)
    test_ratio = cfg.get('test_split', 0.15)
    rows = []
    for label in ['real', 'fake']:
        mel_dir = features_root / 'mel_spectrograms' / label
        for f in mel_dir.glob('*.npy'):
            rows.append({'file': str(f), 'label': 1 if label == 'real' else 0})
    df = pd.DataFrame(rows)
    if df.empty:
        print('No features found to split')
        return
    train_val, test = train_test_split(df, test_size=test_ratio, stratify=df['label'], random_state=42)
    rel_val = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(train_val, test_size=rel_val, stratify=train_val['label'], random_state=42)
    splits_root.mkdir(parents=True, exist_ok=True)
    train.to_csv(splits_root / 'train.csv', index=False)
    val.to_csv(splits_root / 'val.csv', index=False)
    test.to_csv(splits_root / 'test.csv', index=False)
    print('Saved splits to', splits_root)
