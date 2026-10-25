"""t0 — baseline: an MLP on b1's numeric columns and folds."""

import os
from copy import deepcopy

from configs.default_config import cfg as basic_cfg

cfg = deepcopy(basic_cfg)

# Identity
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.output_dir = f"output/{cfg.name}"

# Modules
cfg.dataset = "data.ds_t0"
cfg.model = "models.mdl_t0"
cfg.post_process_pipeline = "postprocess.pp_t0"
cfg.metric = "metrics.metric_b1"

# Model
cfg.hidden = 256
cfg.dropout = 0.1

# Training
cfg.epochs = 10
cfg.batch_size = 256
cfg.lr = 1e-3
