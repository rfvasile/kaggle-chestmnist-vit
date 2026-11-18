"""b1: AutoGluon on the raw columns."""

import os
from copy import deepcopy

from configs.default_config import cfg as basic_cfg
from data.ds_b1 import ID_COL

cfg = deepcopy(basic_cfg)

# Identity
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.output_dir = f"output/{cfg.name}"

# Modules
cfg.dataset = "data.ds_b1"
cfg.post_process_pipeline = "postprocess.pp_b1"
cfg.metric = "metrics.metric_b1"

# The id is not a feature
cfg.ignored_columns = [ID_COL]
cfg.row_id_cols = [ID_COL]
