# %%
"""Config-driven AutoGluon experiment runner."""

import importlib
import importlib.metadata
import json
import shlex
import shutil
import time
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular.predictor.predictor import TabularPredictor

import wandb
from experiment import apply_overrides, check_config_keys, format_override, git_state, to_python
from utils import set_seed


def oof_scores(predictor: TabularPredictor, cfg) -> np.ndarray:
    """Out-of-fold model output, in the shape the postprocess expects."""
    if cfg.positive_class is None:
        return predictor.predict_oof().to_numpy()
    return predictor.predict_proba_oof()[cfg.positive_class].to_numpy()


def test_scores(predictor: TabularPredictor, cfg, rows: pd.DataFrame) -> np.ndarray:
    if cfg.positive_class is None:
        return np.asarray(predictor.predict(rows))
    return predictor.predict_proba(rows)[cfg.positive_class].to_numpy()


# %%
# Input; any other `--key value` overrides the config
parser = ArgumentParser()
parser.add_argument("-C", "--config", default="cfg_b1", help="config name under configs/")
parser.add_argument("--rebuild", action="store_true", help="rebuild the dataset parquets")
args, extra = parser.parse_known_args()

# Config (deepcopy: importlib caches the module)
cfg = deepcopy(importlib.import_module(f"configs.{args.config}").cfg)
check_config_keys(cfg, importlib.import_module("configs.default_config").cfg)
overrides = apply_overrides(cfg, extra)
if args.rebuild:
    cfg.rebuild = True

# Seed
if cfg.seed == -1:
    cfg.seed = int(np.random.randint(1_000_000))

set_seed(cfg.seed)
print(f"Config used: {args.config}  (seed {cfg.seed})")

# Modules
ds = importlib.import_module(cfg.dataset)
post_process_pipeline = importlib.import_module(cfg.post_process_pipeline).post_process_pipeline
sweep_params = importlib.import_module(cfg.post_process_pipeline).sweep_params
write_submission = importlib.import_module(cfg.post_process_pipeline).write_submission
calc_metric = importlib.import_module(cfg.metric).calc_metric

# %%
# Run identity
git = git_state()
started = datetime.now(timezone.utc)
stamp = f"{started:%Y%m%d-%H%M%S}_{git['sha7']}{'-dirty' if git['dirty'] else ''}_seed{cfg.seed}"
run_dir = Path(cfg.output_dir) / stamp
run_dir.mkdir(parents=True, exist_ok=True)
run_id = f"{cfg.name}/{stamp}"

# Tracking
run = wandb.init(
    project=cfg.wandb_project,
    name=stamp,
    tags=[cfg.name],
    notes=cfg.notes,
    mode=cfg.wandb_mode,
    config={**vars(cfg), "git_sha": git["sha"], "git_dirty": git["dirty"], "overrides": overrides},
    settings=wandb.Settings(console="auto", save_code=True),
    reinit="return_previous",
)
# log_code would otherwise take every .py under root, dependencies and older runs' code included
IGNORED_DIRS = {".venv", "output", "wandb", "datamount", "input"}
run.log_code(root=".", exclude_fn=lambda path, _=None: bool(IGNORED_DIRS & set(Path(path).parts)))

print(f"Run id:  {run_id}")
print(f"Run dir: {run_dir}")
print(f"wandb:   {run.url}")
if cfg.notes:
    print(f"Notes:   {cfg.notes}")

# %%
# Data
ds.build(cfg)
train_rows, train_items = ds.load(cfg, "train")
print(f"{len(train_rows)} model rows over {len(train_items)} scored items")

# %%
# Training; groups="fold" bags on the dataset's folds, so OOF is comparable across runs
t0 = time.time()
predictor = TabularPredictor(
    label=cfg.label,
    learner_kwargs={"ignored_columns": cfg.ignored_columns},
    path=str(run_dir / "predictor"),
    problem_type=cfg.problem_type,
    eval_metric=cfg.eval_metric,
    groups="fold",
).fit(
    train_rows,
    presets=cfg.presets,
    hyperparameters=cfg.hyperparameters,
    time_limit=cfg.time_limit,
    num_gpus=cfg.num_gpus,
)
fit_seconds = time.time() - t0
print("--- Training complete ---")

# %%
# Validation: sweep the decoder grid on OOF, ranked by the competition metric
oof = oof_scores(predictor, cfg)

# Sweep
grid = list(sweep_params(cfg))
results = [
    {**params, **calc_metric(cfg, post_process_pipeline(cfg, train_rows, oof, train_items, **params))}
    for params in grid
]

# Best params
results_df = pd.DataFrame(results).sort_values(cfg.selection_metric, ascending=False)
best = results_df.iloc[0]
best_params = {k: to_python(best[k]) for k in grid[0]}
print(results_df.to_string(index=False))
print(f"\nBest OOF {cfg.selection_metric}: {best[cfg.selection_metric]:.4f}  {best_params}")

# %%
# Artifacts
best_pred = post_process_pipeline(cfg, train_rows, oof, train_items, **best_params)
best_pred.to_csv(run_dir / "oof.csv", index=False)
train_rows[cfg.row_id_cols].assign(prob=oof).to_parquet(run_dir / "oof_probs.parquet", index=False)
results_df.to_csv(run_dir / "sweep.csv", index=False)
predictor.leaderboard().to_csv(run_dir / "leaderboard.csv", index=False)

with open(run_dir / "fit_info.json", "w") as f:
    json.dump(predictor.info(), f, indent=2, default=str)

# %%
# Refit on all the data
submission_model = predictor.model_best
if cfg.refit_full:
    refit_map = predictor.refit_full()
    submission_model = predictor.model_best
    print(f"refit on full data: {len(refit_map)} model(s); submission uses {submission_model}")

# %%
# Submission
if cfg.test:
    test_rows, test_items = ds.load(cfg, "test")
    probs = test_scores(predictor, cfg, test_rows)
    predicted = post_process_pipeline(cfg, test_rows, probs, test_items, **best_params)
    submission = write_submission(cfg, predicted, run_dir / "submission.csv")
    test_rows[cfg.row_id_cols].assign(prob=probs).to_parquet(run_dir / "test_probs.parquet", index=False)
    shutil.copy2(run_dir / "submission.csv", "submission.csv")
    print(f"wrote {run_dir}/submission.csv ({len(submission)} rows)")

# %%
# Record the run; run.json keeps what an offline reader needs
predictor_bytes = sum(f.stat().st_size for f in (run_dir / "predictor").rglob("*") if f.is_file())
cv = calc_metric(cfg, best_pred)

# W&B summary and tables
run.summary.update(
    {
        **cv,
        **{f"best_{k}": v for k, v in best_params.items()},
        "fit_seconds": round(fit_seconds, 1),
        "predictor_bytes": predictor_bytes,
        "submission_model": submission_model,
        "refit_full": cfg.refit_full,
        "rows": len(train_rows),
        "items": len(train_items),
        "autogluon": importlib.metadata.version("autogluon.tabular"),
    }
)
run.log({"sweep": wandb.Table(dataframe=results_df), "leaderboard": wandb.Table(dataframe=predictor.leaderboard())})

# The probabilities, not the predictor
probs_artifact = wandb.Artifact(f"probs-{cfg.name}", type="predictions")
for name in ("oof_probs.parquet", "test_probs.parquet"):
    if (run_dir / name).exists():
        probs_artifact.add_file(str(run_dir / name), name=name)
run.log_artifact(probs_artifact)

# run.json
manifest = {
    "run_id": run_id,
    "config": args.config,
    "notes": cfg.notes,
    "wandb": {"id": run.id, "url": run.url, "project": cfg.wandb_project},
    "best_params": best_params,
    "cv": cv,
    "fit_seconds": round(fit_seconds, 1),
    "submission_model": submission_model,
    "refit_full": cfg.refit_full,
}
with open(run_dir / "run.json", "w") as f:
    json.dump(manifest, f, indent=2, default=str)

# Drop the predictor unless kept
if not cfg.keep_predictor:
    shutil.rmtree(run_dir / "predictor")
    print(f"dropped predictor ({predictor_bytes / 1e9:.1f} GB); set cfg.keep_predictor to keep it")

run.finish()

# Reproduce line
print(f"\n{run_id}  {cfg.selection_metric}={cv[cfg.selection_metric]:.4f}")
replay = [f"--{k} {shlex.quote(format_override(v))}" for k, v in overrides.items() if k != "seed"]
runner = Path(__file__).name
print(f"reproduce: python {runner} -C {args.config} --seed {cfg.seed} {' '.join(replay)}".rstrip())
print(f"           (from {git['describe']})")
