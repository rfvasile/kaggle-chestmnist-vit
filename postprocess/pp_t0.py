"""t0 decoder: the sigmoid of the logit is the prediction."""

import numpy as np
import pandas as pd
import torch

from postprocess.pp_b1 import sweep_params, write_submission

__all__ = ["post_process_pipeline", "sweep_params", "write_submission"]


def post_process_pipeline(cfg, data: dict, df: pd.DataFrame) -> pd.DataFrame:
    """One prediction per entry of data["idx"], a row position in df."""
    out = df.reset_index(drop=True).iloc[np.asarray(data["idx"])].copy()
    out["pred"] = torch.sigmoid(torch.as_tensor(data["logits"])).numpy()
    return out
