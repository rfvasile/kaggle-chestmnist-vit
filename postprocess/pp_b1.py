"""b1 decoder: the positive-class probability is the prediction."""

import pandas as pd

from data.ds_b1 import ID_COL


def sweep_params(cfg):
    """The decoder grid for one run, as dicts of post_process_pipeline kwargs; nothing to sweep here."""
    yield {}


def post_process_pipeline(cfg, row_df: pd.DataFrame, probs, item_df: pd.DataFrame) -> pd.DataFrame:
    """One prediction per item; rows are items here, so `probs` lines up with item_df."""
    return item_df.assign(pred=probs)


def write_submission(cfg, pred_df: pd.DataFrame, path) -> pd.DataFrame:
    """Write this competition's submission file: id plus each label's probability."""
    preds = [f"pred_{label}" for label in cfg.labels]
    out = pred_df[[ID_COL, *preds]].rename(columns=dict(zip(preds, cfg.labels)))
    out.to_csv(path, index=False)
    return out
