"""t0 decoder: the sigmoid of each label's logit is its prediction."""

import numpy as np
import pandas as pd
import torch

from data.ds_b1 import ID_COL


def sweep_params(cfg):
    """The decoder grid for one run, as dicts of post_process_pipeline kwargs; nothing to sweep here."""
    yield {}


def post_process_pipeline(cfg, data: dict, df: pd.DataFrame) -> pd.DataFrame:
    """One row per entry of data["idx"], a row position in df, with a pred_<label> column per label."""
    out = df.reset_index(drop=True).iloc[np.asarray(data["idx"])].copy()
    probs = torch.sigmoid(torch.as_tensor(data["logits"])).numpy()
    return out.assign(**{f"pred_{label}": probs[:, i] for i, label in enumerate(cfg.labels)})


def write_submission(cfg, pred_df: pd.DataFrame, path) -> pd.DataFrame:
    """Write this competition's submission file: id plus each label's probability."""
    preds = [f"pred_{label}" for label in cfg.labels]
    out = pred_df[[ID_COL, *preds]].rename(columns=dict(zip(preds, cfg.labels)))
    out.to_csv(path, index=False)
    return out
