import numpy as np
import pandas as pd
import pytest
import torch

from configs.cfg_t0 import cfg
from data import ds_b1, ds_t0


def test_each_item_pairs_its_rows_image_with_its_labels(monkeypatch):
    # Image i is filled with the value i, so an item shows which image it got
    images = np.broadcast_to(np.arange(4, dtype=np.uint8)[:, None, None], (4, 64, 64))
    monkeypatch.setattr(ds_b1, "images", lambda cfg, split: images)
    labels = np.random.default_rng(0).integers(0, 2, (3, len(cfg.labels)))
    df = pd.DataFrame(labels, columns=cfg.labels).assign(img_idx=[2, 0, 3])

    dataset = ds_t0.CustomDataset(df, cfg, mode="train")
    for k, img_idx in enumerate(df["img_idx"]):
        item = dataset[k]
        assert item["image"].shape == (1, 64, 64)
        assert item["image"].unique().item() == pytest.approx(img_idx / 127.5 - 1), f"row {k} got the wrong image"
        assert torch.equal(item["target"], torch.tensor(labels[k], dtype=torch.float32))
