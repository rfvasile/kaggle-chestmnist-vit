"""t0 decoder: the sigmoid of each label's logit is its prediction."""

import numpy as np
import pandas as pd
import torch

from postprocess.pp_b1 import sweep_params, write_submission

__all__ = ["post_process_pipeline", "sweep_params", "write_submission"]


def post_process_pipeline(cfg, data: dict, df: pd.DataFrame) -> pd.DataFrame:
    """One row per entry of data["idx"], a row position in df, with a pred_<label> column per label."""
    out = df.reset_index(drop=True).iloc[np.asarray(data["idx"])].copy()
    probs = torch.sigmoid(torch.as_tensor(data["logits"])).numpy()
    return out.assign(**{f"pred_{label}": probs[:, i] for i, label in enumerate(cfg.labels)})
