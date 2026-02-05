from pathlib import Path
from src.train import train_model

CFG = Path('config/config.yaml')
TRAIN_CSV = Path('data/splits/train.csv')
VAL_CSV = Path('data/splits/val.csv')
MODEL_OUT = Path('models/best_model.pth')

print('Starting training')
train_model(CFG, TRAIN_CSV, VAL_CSV, MODEL_OUT)
print('Training finished')
