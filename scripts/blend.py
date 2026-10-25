"""Average the saved outputs of several train_nn.py runs and score the average as one prediction.

    python scripts/blend.py -C cfg_t0 --name t0 --keys logits --runs output/cfg_t0/<run> output/cfg_t0/<run>

Decodes and scores with config `-C`'s modules; any other `--key value` overrides that config.
"""

import importlib
import json
import shutil
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import torch

from experiment import apply_overrides, git_state

# Input
parser = ArgumentParser()
parser.add_argument("-C", "--config", required=True, help="config whose dataset, postprocess and metric to use")
parser.add_argument("--name", required=True, help="names the output directory, output/blend_<name>")
parser.add_argument("--keys", nargs="+", required=True, help="output keys to average")
parser.add_argument("--runs", nargs="+", required=True, help="run directories")
parser.add_argument("--test_only", action="store_true", help="skip the OOF score; the runs have only test outputs")
args, extra = parser.parse_known_args()

# Config and modules
cfg = deepcopy(importlib.import_module(f"configs.{args.config}").cfg)
overrides = apply_overrides(cfg, extra)
ds = importlib.import_module(cfg.dataset)
pp = importlib.import_module(cfg.post_process_pipeline)
calc_metric = importlib.import_module(cfg.metric).calc_metric
id_col = ds.ID_COL


def averaged(split_df, filename: str) -> dict:
    """The mean of `--keys` over every run, in split_df's row order."""
    total = None
    for run in args.runs:
        saved = torch.load(Path(run) / filename, weights_only=False)
        order = {tid: k for k, tid in enumerate(saved[id_col])}
        rows = torch.as_tensor([order[tid] for tid in split_df[id_col]])
        part = {k: saved[k][rows] for k in args.keys}
        total = part if total is None else {k: total[k] + part[k] for k in args.keys}
    return {**{k: v / len(args.runs) for k, v in total.items()}, "idx": torch.arange(len(split_df))}


# Run identity
params = next(iter(pp.sweep_params(cfg)))
git = git_state(required=False)
stamp = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}_{git['sha7']}{'-dirty' if git['dirty'] else ''}"
run_dir = Path(f"output/blend_{args.name}") / stamp
run_dir.mkdir(parents=True, exist_ok=True)

# OOF score
cv = {}
if not args.test_only:
    train_df = ds.load(cfg, "train")
    oof_pred = pp.post_process_pipeline(cfg, averaged(train_df, "oof_outputs.pt"), train_df, **params)
    cv = calc_metric(cfg, oof_pred)
    oof_pred.to_csv(run_dir / "oof.csv", index=False)

# Submission
test_df = ds.load(cfg, "test")
test_pred = pp.post_process_pipeline(cfg, averaged(test_df, "test_outputs.pt"), test_df, **params)
pp.write_submission(cfg, test_pred, run_dir / "submission.csv")
shutil.copy2(run_dir / "submission.csv", "submission.csv")

# Record the run
manifest = {
    "config": args.config,
    "overrides": overrides,
    "keys": args.keys,
    "runs": args.runs,
    "params": params,
    "cv": cv,
    "git": git,
}
(run_dir / "run.json").write_text(json.dumps(manifest, indent=2, default=str))
print(f"{run_dir}  " + ("  ".join(f"{k}={v:.4f}" for k, v in cv.items()) or f"test only, {len(test_pred)} rows"))
