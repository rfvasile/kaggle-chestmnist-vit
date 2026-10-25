"""t0 dataset: ds_b1's rows and folds, the numeric columns as one float tensor."""

from copy import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, default_collate

from data import ds_b1
from data.ds_b1 import ID_COL


def b1_cfg(cfg):
    """cfg as ds_b1 needs to see it, so its parquets and folds are reused rather than rebuilt."""
    b1 = copy(cfg)
    b1.dataset = "data.ds_b1"
    return b1


def load(cfg, split: str = "train") -> pd.DataFrame:
    b1 = b1_cfg(cfg)
    ds_b1.build(b1)
    return ds_b1.load(b1, split)[0]


def features(cfg, df: pd.DataFrame) -> np.ndarray:
    """Numeric columns bar id, label and fold; signed log1p evens their scales without train statistics."""
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
