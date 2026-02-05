from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model_path: Path, test_csv: Path, device: str = 'cpu'):
    from src.data.dataset import AudioFeatureDataset
    from src.model.cnn import DeepCNN

    ds = AudioFeatureDataset(str(test_csv))
    loader = torch.utils.data.DataLoader(ds, batch_size=32)
    dev = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model = DeepCNN().to(dev)
    model.load_state_dict(torch.load(model_path, map_location=dev))
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(dev)
            out = model(xb)
            preds = (out.cpu().numpy().squeeze() >= 0.5).astype(int)
            y_pred.extend(preds.tolist())
            y_true.extend(yb.numpy().astype(int).tolist())
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(f'Accuracy={acc:.4f} Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f}')
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    out_dir = Path('visualizations')
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / 'confusion_matrix.png')
