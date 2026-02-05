from pathlib import Path
from src.preprocessing.preprocess import run_preprocessing
from src.utils.balance_data import check_and_balance
from src.features.extract_features import extract_and_save
from src.utils.create_splits import create_stratified_splits
from src.train import train_model
from src.evaluate import evaluate_model
import argparse

PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = Path('ml-service/data/Audio_Dataset') if Path('ml-service/data/Audio_Dataset').exists() else Path('ml-service/data/audio_dataset')
PROCESSED_ROOT = PROJECT_ROOT / 'data'
FEATURES_ROOT = PROJECT_ROOT / 'data' / 'features'
SPLITS_ROOT = PROJECT_ROOT / 'data' / 'splits'
MODEL_OUT = PROJECT_ROOT / 'models' / 'best_model.pth'
CONFIG = PROJECT_ROOT / 'config' / 'config.yaml'


def run_all():
    print('Starting pipeline...')
    run_preprocessing(DATA_ROOT, PROJECT_ROOT / 'data', CONFIG)
    check_and_balance(PROJECT_ROOT / 'data', CONFIG)
    extract_and_save(PROJECT_ROOT / 'data', PROJECT_ROOT / 'data' / 'features', CONFIG)
    create_stratified_splits(PROJECT_ROOT / 'data' / 'features', PROJECT_ROOT / 'data' / 'splits', CONFIG)
    train_model(CONFIG, PROJECT_ROOT / 'data' / 'splits' / 'train.csv', PROJECT_ROOT / 'data' / 'splits' / 'val.csv', MODEL_OUT)
    evaluate_model(MODEL_OUT, PROJECT_ROOT / 'data' / 'splits' / 'test.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', help='Run full pipeline')
    args = parser.parse_args()
    if args.run:
        run_all()
    else:
        print('Use --run to execute the full pipeline')
