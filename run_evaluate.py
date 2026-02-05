from pathlib import Path
from src.evaluate import evaluate_model

MODEL = Path('models/best_model.pth')
TEST_CSV = Path('data/splits/test.csv')

print('Starting evaluation')
evaluate_model(MODEL, TEST_CSV)
print('Evaluation complete')
