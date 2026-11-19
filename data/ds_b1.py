"""b1 dataset: the prepared rows, as train_gluon.py reads them."""

import pandas as pd

from data import prepare

ID_COL = prepare.ID_COL  # the runners read ds.ID_COL

build = prepare.build


def load(cfg, split: str = "train") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(row_df, item_df); one row per item here, so the same frame twice."""
    df = prepare.load(cfg, split)
    return df, df
