"""b1 dataset: the raw CSVs, with stratified folds."""

from pathlib import Path

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import StratifiedKFold

ID_COL = "<id>"

SPLITS = ("train", "test")


def data_dir(cfg) -> Path:
    """Keyed on the fold seed, so splits never overwrite each other."""
    return Path(cfg.datamount) / f"{cfg.dataset.split('.')[-1]}_f{cfg.fold_seed}"


def assign_folds(cfg, df: pd.DataFrame) -> pd.DataFrame:
    """Stratified folds from cfg.fold_seed, shared by every model seed."""
    out = df.copy()
    out["fold"] = -1
    kf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.fold_seed)
    for fold, (_, val_idx) in enumerate(kf.split(out, out[cfg.label])):
        out.iloc[val_idx, out.columns.get_loc("fold")] = fold
    return out


def build(cfg) -> None:
    """Raw CSVs -> parquets; a no-op once built, unless cfg.rebuild."""
    # Skip when already built
    out = data_dir(cfg)
    if all((out / f"{split}.parquet").exists() for split in SPLITS) and not cfg.rebuild:
        print(f"dataset already built at {out}")
        return

    # Download if missing
    raw = Path(cfg.raw_dir)
    if not raw.exists():
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files(cfg.competition, path=cfg.raw_dir, unzip=True)

    # Splits, train with folds
    out.mkdir(parents=True, exist_ok=True)
    train = assign_folds(cfg, pd.read_csv(raw / "train.csv"))
    train.to_parquet(out / "train.parquet", index=False)
    pd.read_csv(raw / "test.csv").to_parquet(out / "test.parquet", index=False)
    print(f"built {out}: {len(train)} train rows")


def load(cfg, split: str = "train") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(row_df, item_df); one row per item here, so the same frame twice."""
    df = pd.read_parquet(data_dir(cfg) / f"{split}.parquet")
    return df, df
