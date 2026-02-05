from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class AudioFeatureDataset(Dataset):
    """Loads precomputed feature numpy files and returns tensors."""

    def __init__(self, csv_file: str):
        import pandas as pd
        self.df = pd.read_csv(csv_file)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        arr = np.load(row['file'])
        if arr.ndim == 2:
            arr = np.expand_dims(arr, 0)
        tensor = torch.tensor(arr, dtype=torch.float32)
        label = torch.tensor(int(row['label']), dtype=torch.float32)
        return tensor, label
