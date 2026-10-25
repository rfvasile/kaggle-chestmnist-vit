"""b1 — baseline: AutoGluon's medium portfolio on the raw columns."""

import os
from copy import deepcopy

from configs.default_config import cfg as basic_cfg

cfg = deepcopy(basic_cfg)

# Identity
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.output_dir = f"output/{cfg.name}"

# Modules
cfg.dataset = "data.ds_b1"
cfg.post_process_pipeline = "postprocess.pp_b1"
cfg.metric = "metrics.metric_b1"

# The id names a row; it is not a feature
cfg.ignored_columns = ["id"]
cfg.row_id_cols = ["id"]
