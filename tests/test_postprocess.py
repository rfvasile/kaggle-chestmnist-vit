import numpy as np
import pandas as pd
import torch

from configs.default_config import cfg
from metrics.metric_b1 import calc_metric
from postprocess import pp_t0


def test_decoded_logits_score_and_submit_in_label_order(tmp_path):
    y = np.random.default_rng(0).integers(0, 2, (50, len(cfg.labels)))
    df = pd.DataFrame(y, columns=cfg.labels).assign(id=[f"tst_{i:06d}" for i in range(len(y))])
    data = {"idx": torch.arange(len(y)), "logits": torch.as_tensor(2.0 * y - 1)}

    pred_df = pp_t0.post_process_pipeline(cfg, data, df)
    assert calc_metric(cfg, pred_df)["auc"] == 1.0, "a logit per label should rank its own label perfectly"

    submission = pp_t0.write_submission(cfg, pred_df, tmp_path / "submission.csv")
    assert list(submission.columns) == ["id", *cfg.labels]
