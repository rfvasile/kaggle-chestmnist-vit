"""Competition metric: mean column-wise ROC AUC over cfg.labels."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def calc_metric(cfg, pred_df: pd.DataFrame) -> dict[str, float]:
    """pred_df holds each label and its prediction as pred_<label>; returns the mean AUC and each label's."""
    per_label = {f"auc_{label}": float(roc_auc_score(pred_df[label], pred_df[f"pred_{label}"])) for label in cfg.labels}
    return {"auc": float(np.mean(list(per_label.values()))), **per_label}
