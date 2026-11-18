# %%
"""Test predictions from a train_nn.py run's fold checkpoints."""

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
args, extra = parser.parse_known_args()

# Config: the run's, plus any `--key value`
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
test_sum = None
for path in checkpoints:
    model = Net(dataset=test_dataset, cfg=cfg, mode="test").to(cfg.device)
    model.load_state_dict(torch.load(path, map_location=cfg.device, weights_only=True)["model"])
    data, _ = predict(model, loader, ds.batch_to_device, cfg.device, cfg.mixed_precision, cfg.amp_dtype)
    test_sum = data if test_sum is None else {k: v if k == "idx" else test_sum[k] + v for k, v in data.items()}
    print(f"predicted {path.name}")
    del model
    torch.cuda.empty_cache()

test_data = {k: v if k == "idx" else v / len(checkpoints) for k, v in test_sum.items()}

# Artifacts
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
torch.save({id_col: test_df[id_col].tolist(), **test_data} if id_col else test_data, out / "test_outputs.pt")
predicted = pp.post_process_pipeline(cfg, test_data, test_df, **manifest.get("best_params", {}))
pp.write_submission(cfg, predicted, out / "submission.csv")
(out / "run.json").write_text(
    json.dumps({"config": manifest["config"], "overrides": overrides, "inferred_from": str(args.run)}, indent=2)
)
print(f"wrote {out}/test_outputs.pt and submission.csv ({len(test_df)} rows)")
