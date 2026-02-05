import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from yaml import safe_load
from tqdm import tqdm
import numpy as np
import random




def load_config(path: Path) -> dict:
    with open(path, 'r') as f:
        return safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(cfg_path: Path, train_csv: Path, val_csv: Path, model_out: Path):
    cfg = load_config(cfg_path)
    set_seed(cfg.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.get('device', 'auto') == 'auto' else 'cpu')
    from src.data.dataset import AudioFeatureDataset
    from src.model.cnn import DeepCNN

    train_ds = AudioFeatureDataset(str(train_csv))
    val_ds = AudioFeatureDataset(str(val_csv))
    train_loader = DataLoader(train_ds, batch_size=cfg.get('batch_size', 32), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.get('batch_size', 32), shuffle=False)

    model = DeepCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.get('lr', 0.001))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(Path('logs') / 'tensorboard'))
    except Exception:
        writer = None
    best_val = 0.0
    patience = cfg.get('patience', 7)
    no_improve = 0

    for epoch in range(cfg.get('epochs', 50)):
        model.train()
        train_losses = []
        for xb, yb in tqdm(train_loader, desc=f'Epoch {epoch} train'):
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train = sum(train_losses) / max(1, len(train_losses))

        model.eval()
        val_losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb.unsqueeze(1))
                val_losses.append(loss.item())
                preds = (pred.detach().cpu().squeeze() >= 0.5).numpy()
                correct += (preds == yb.cpu().numpy()).sum()
                total += len(yb)
        avg_val = sum(val_losses) / max(1, len(val_losses))
        acc = correct / max(1, total)
        if writer:
            writer.add_scalar('Loss/train', avg_train, epoch)
            writer.add_scalar('Loss/val', avg_val, epoch)
            writer.add_scalar('Acc/val', acc, epoch)
        scheduler.step(avg_val)

        print(f'Epoch {epoch} train_loss={avg_train:.4f} val_loss={avg_val:.4f} val_acc={acc:.4f}')

        if acc > best_val:
            best_val = acc
            torch.save(model.state_dict(), model_out)
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print('Early stopping')
            break

    if writer:
        writer.close()
