import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from configs.default_config import cfg
from metrics.metric_b1 import calc_metric


def test_auc_is_the_mean_columnwise_auc():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, (200, len(cfg.labels)))
    p = 0.3 * y + rng.random(y.shape)
    pred_df = pd.DataFrame(y, columns=cfg.labels).join(pd.DataFrame(p, columns=[f"pred_{c}" for c in cfg.labels]))
    assert calc_metric(cfg, pred_df)["auc"] == pytest.approx(roc_auc_score(y, p, average="macro"))
