"""t0 dataset: ds_b1's rows and folds, each image scaled to [-1, 1] with its 14 labels as the target."""

from copy import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, default_collate

from data import ds_b1

ID_COL = ds_b1.ID_COL  # train_nn.py keys saved outputs on it


def b1_cfg(cfg):
    """cfg as ds_b1 needs to see it, so its arrays and folds are reused rather than rebuilt."""
    b1 = copy(cfg)
    b1.dataset = "data.ds_b1"
    return b1


def load(cfg, split: str = "train") -> pd.DataFrame:
    b1 = b1_cfg(cfg)
    ds_b1.build(b1)
    return ds_b1.load(b1, split)


# Torch dataset
class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg, mode: str = "train"):
        # This split's rows only, read into memory as uint8
        images = ds_b1.images(b1_cfg(cfg), "test" if mode == "test" else "train")
        self.images = torch.from_numpy(np.ascontiguousarray(images[df["img_idx"].to_numpy()]))
        self.labels = None
        if set(cfg.labels) <= set(df.columns):
            self.labels = torch.tensor(df[cfg.labels].to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].unsqueeze(0).float() / 127.5 - 1
        item = {"image": image, "idx": torch.tensor(idx, dtype=torch.long)}
        if self.labels is not None:
            item["target"] = self.labels[idx]
        return item


# Batching
tr_collate_fn = default_collate
val_collate_fn = default_collate


def batch_to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
