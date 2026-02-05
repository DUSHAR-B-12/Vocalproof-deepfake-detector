from pathlib import Path
from src.data.dataset import AudioFeatureDataset
import torch

TEST_CSV = Path('data/splits/train.csv')
ds = AudioFeatureDataset(str(TEST_CSV))
print(f"Dataset length: {len(ds)}")

for i in range(3):
    x, y = ds[i]
    print(f"Sample {i}: x.shape={x.shape}, x.ndim={x.ndim}, y={y.item()}")
    
loader = torch.utils.data.DataLoader(ds, batch_size=2)
for xb, yb in loader:
    print(f"Batch: xb.shape={xb.shape}, yb.shape={yb.shape}")
    break
