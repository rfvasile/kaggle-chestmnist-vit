# %%
"""Inference with a trained train_nn.py run.

    python infer_nn.py --run output/cfg_t0/<run> --out output/infer/t0
    python infer_nn.py --run <dir> --out <dir> --raw_dir /kaggle/input/<comp>
    python infer_nn.py --run <dir> --out <dir> --per_fold

The config comes from the run's run.json plus any `--key value`.
"""

import importlib
import json
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiment import apply_overrides, load_run_config
from utils import predict

# Input
parser = ArgumentParser()
parser.add_argument("--run", required=True, help="a train_nn.py run directory with run.json and fold*.pth")
parser.add_argument("--out", required=True, help="where to write test_outputs.pt, run.json, submission.csv")
parser.add_argument("--per_fold", action="store_true", help="also write test_fold_outputs.pt, for per-model votes")
args, extra = parser.parse_known_args()

# Config: the run's, plus any `--key value` given here
manifest = json.loads((Path(args.run) / "run.json").read_text())
cfg = load_run_config(args.run)
overrides = {**manifest.get("overrides", {}), **apply_overrides(cfg, extra)}
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

# Modules
ds = importlib.import_module(cfg.dataset)
Net = importlib.import_module(cfg.model).Net
pp = importlib.import_module(cfg.post_process_pipeline)
id_col = getattr(ds, "ID_COL", None)

# Data
test_df = ds.load(cfg, "test")
test_dataset = ds.CustomDataset(test_df, cfg=cfg, mode="test")
loader = DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=cfg.batch_size,
    num_workers=cfg.num_workers,
    pin_memory=cfg.pin_memory,
    collate_fn=ds.val_collate_fn,
)

# Checkpoints
checkpoints = [Path(args.run) / f"fold{fold}.pth" for fold in manifest["folds"]]
missing = [str(c) for c in checkpoints if not c.exists()]
assert not missing, f"no checkpoint for {missing}; train with --save_checkpoint True"

# Predict with each fold model, then average
test_sum, fold_outputs = None, []
for path in checkpoints:
    model = Net(dataset=test_dataset, cfg=cfg, mode="test").to(cfg.device)
    model.load_state_dict(torch.load(path, map_location=cfg.device, weights_only=True)["model"])
    data, _ = predict(model, loader, ds.batch_to_device, cfg.device, cfg.mixed_precision, cfg.amp_dtype)
    if args.per_fold:
        fold_outputs.append(data)
    test_sum = data if test_sum is None else {k: v if k == "idx" else test_sum[k] + v for k, v in data.items()}
    print(f"predicted {path.name}")
    del model
    torch.cuda.empty_cache()

test_data = {k: v if k == "idx" else v / len(checkpoints) for k, v in test_sum.items()}

# Artifacts
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
torch.save({id_col: test_df[id_col].tolist(), **test_data} if id_col else test_data, out / "test_outputs.pt")

# Each fold's outputs, stacked on a leading fold axis for a vote; no submission
if args.per_fold:
    stacked = {k: torch.stack([d[k] for d in fold_outputs]) for k in fold_outputs[0] if k != "idx"}
    stacked["idx"] = test_data["idx"]
    torch.save({id_col: test_df[id_col].tolist(), **stacked} if id_col else stacked, out / "test_fold_outputs.pt")
else:
    predicted = pp.post_process_pipeline(cfg, test_data, test_df, **manifest.get("best_params", {}))
    pp.write_submission(cfg, predicted, out / "submission.csv")
(out / "run.json").write_text(
    json.dumps({"config": manifest["config"], "overrides": overrides, "inferred_from": str(args.run)}, indent=2)
)
written = "test_fold_outputs.pt" if args.per_fold else "submission.csv"
print(f"wrote {out}/test_outputs.pt and {written} ({len(test_df)} rows)")
