"""Competition metric: ROC AUC of the predicted probability."""

import pandas as pd
from sklearn.metrics import roc_auc_score


def calc_metric(cfg, pred_df: pd.DataFrame) -> dict[str, float]:
    return {"auc": float(roc_auc_score(pred_df[cfg.label], pred_df["pred"]))}
