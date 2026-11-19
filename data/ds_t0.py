"""t0 dataset: the prepared numeric columns as one float tensor."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, default_collate

from data import prepare
from data.prepare import ID_COL  # the runners read ds.ID_COL


def load(cfg, split: str = "train") -> pd.DataFrame:
    prepare.build(cfg)
    return prepare.load(cfg, split)


def features(cfg, df: pd.DataFrame) -> np.ndarray:
    """Numeric features; signed log1p evens scales without train statistics."""
    x = df.drop(columns=[ID_COL, cfg.label, "fold"], errors="ignore").select_dtypes("number")
    x = x.to_numpy(dtype=np.float32)
    return np.nan_to_num(np.sign(x) * np.log1p(np.abs(x)))


# Torch dataset
class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg, mode: str = "train"):
        self.features = torch.from_numpy(features(cfg, df))
        self.labels = None
        if cfg.label in df.columns:
            self.labels = torch.tensor(df[cfg.label].to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = {"features": self.features[idx], "idx": torch.tensor(idx, dtype=torch.long)}
        if self.labels is not None:
            item["target"] = self.labels[idx]
        return item


# Batching
tr_collate_fn = default_collate
val_collate_fn = default_collate


def batch_to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
