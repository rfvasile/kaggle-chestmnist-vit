"""b1 dataset: PNGs stacked into uint8 arrays, folds stratified on all labels."""

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image

ID_COL = "id"


def data_dirs(cfg) -> tuple[Path, Path]:
    """(fold_dir, image_dir): folds keyed on the fold seed, images shared across seeds."""
    name = cfg.dataset.split(".")[-1]
    return Path(cfg.datamount) / f"{name}_f{cfg.fold_seed}", Path(cfg.datamount) / name


def assign_folds(cfg, df: pd.DataFrame) -> pd.DataFrame:
    """Folds stratified on cfg.labels, drawn from cfg.fold_seed so every model seed shares them."""
    out = df.copy()
    out["fold"] = -1
    kf = MultilabelStratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.fold_seed)
    for fold, (_, val_idx) in enumerate(kf.split(out, out[cfg.labels].to_numpy())):
        out.iloc[val_idx, out.columns.get_loc("fold")] = fold
    return out


def read_images(img_dir: Path, ids: pd.Series) -> np.ndarray:
    """(N, 64, 64) uint8 array of images, in the order of `ids`."""
    return np.stack([np.asarray(Image.open(img_dir / f"{i}.png")) for i in ids])


def build(cfg) -> None:
    """Raw CSVs and PNGs -> image arrays, plus folds for cfg.fold_seed. Each a no-op once built, unless cfg.rebuild."""
    fold_dir, image_dir = data_dirs(cfg)
    raw = Path(cfg.raw_dir)

    # Download competition data if not present
    if not raw.exists():
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files(cfg.competition, path=cfg.raw_dir, unzip=True)

    # Images and test ids, shared by every fold seed; img_idx is a row's index into its .npy
    if cfg.rebuild or not (image_dir / "test.parquet").exists():
        image_dir.mkdir(parents=True, exist_ok=True)
        train_ids = pd.read_csv(raw / "train_labels.csv")[ID_COL]
        test = pd.read_csv(raw / "sample_submission.csv")[[ID_COL]]
        np.save(image_dir / "train_images.npy", read_images(raw / "train" / "train", train_ids))
        np.save(image_dir / "test_images.npy", read_images(raw / "test" / "test", test[ID_COL]))
        test.assign(img_idx=np.arange(len(test))).to_parquet(image_dir / "test.parquet", index=False)
        print(f"built {image_dir}: {len(train_ids)} train / {len(test)} test images")
    else:
        print(f"images already built at {image_dir}")

    # Folds for this seed
    if cfg.rebuild or not (fold_dir / "train.parquet").exists():
        fold_dir.mkdir(parents=True, exist_ok=True)
        train = pd.read_csv(raw / "train_labels.csv")
        train = assign_folds(cfg, train.assign(img_idx=np.arange(len(train))))
        train.to_parquet(fold_dir / "train.parquet", index=False)
        print(f"built {fold_dir}: {len(train)} train rows")
    else:
        print(f"folds already built at {fold_dir}")


def load(cfg, split: str = "train") -> pd.DataFrame:
    """One row per image: id and img_idx, plus labels and fold for train."""
    fold_dir, image_dir = data_dirs(cfg)
    return pd.read_parquet((fold_dir if split == "train" else image_dir) / f"{split}.parquet")


def images(cfg, split: str) -> np.ndarray:
    """(N, 64, 64) uint8, memory-mapped; index with load(cfg, split)["img_idx"]."""
    return np.load(data_dirs(cfg)[1] / f"{split}_images.npy", mmap_mode="r")


# %%
if __name__ == "__main__":
    from copy import deepcopy
    from importlib import import_module

    build(deepcopy(import_module("configs.cfg_b1").cfg))
