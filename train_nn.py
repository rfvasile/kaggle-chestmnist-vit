# %%
"""Config-driven torch experiment runner."""

import gc
import importlib
import json
import shlex
import shutil
import sys
import time
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler
from torch.amp.autocast_mode import autocast
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, get_swa_multi_avg_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

import wandb
from experiment import apply_overrides, check_config_keys, format_override, git_state, to_python
from utils import calc_grad_norm, predict, set_seed

# %autoreload for REPL sessions only; in a script it would hot-patch a running fit
if hasattr(sys, "ps1") or "ipykernel" in sys.modules:
    from jurigged import watch

    watch(".")

SCHEDULES = {"linear": get_linear_schedule_with_warmup, "cosine": get_cosine_schedule_with_warmup}


def param_groups(model, weight_decay: float) -> list[dict]:
    """No weight decay on biases and norm weights, the usual transformer recipe."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim < 2 else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def swa_average(averaged: list, current: list, n_averaged) -> None:
    """torch's SWA mean for float tensors; integer buffers (ids, BatchNorm counters) follow the model."""
    if averaged[0].is_floating_point():
        get_swa_multi_avg_fn()(averaged, current, n_averaged)
    else:
        for a, c in zip(averaged, current):
            a.copy_(c)


def scatter(parts: list[tuple[np.ndarray, dict]], n: int) -> dict:
    """Per-fold outputs -> one dict over n rows, each row at the position given with its fold."""
    keys = [k for k in parts[0][1] if k != "idx"]
    out = {k: torch.zeros((n, *parts[0][1][k].shape[1:])) for k in keys}
    for positions, data in parts:
        rows = torch.as_tensor(positions[data["idx"].numpy()])
        for k in keys:
            out[k][rows] = data[k]
    out["idx"] = torch.arange(n)
    return out


# %%
# Input; any other `--key value` overrides the config
parser = ArgumentParser()
parser.add_argument("-C", "--config", default="cfg_t0", help="config name under configs/")
args, extra = parser.parse_known_args()

# Config (deepcopy: importlib caches the module)
cfg = deepcopy(importlib.import_module(f"configs.{args.config}").cfg)
check_config_keys(cfg, importlib.import_module("configs.default_config").cfg)
overrides = apply_overrides(cfg, extra)
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

# Seed; each fold reseeds with cfg.seed + fold, so it trains the same when run alone
if cfg.seed == -1:
    cfg.seed = int(np.random.randint(1_000_000))
print(f"Config used: {args.config}  (seed {cfg.seed})")

# Modules
ds = importlib.import_module(cfg.dataset)
Net = importlib.import_module(cfg.model).Net
CustomDataset, tr_collate_fn, val_collate_fn, batch_to_device = (
    ds.CustomDataset,
    ds.tr_collate_fn,
    ds.val_collate_fn,
    ds.batch_to_device,
)
id_col = getattr(ds, "ID_COL", None)
train_batch_sampler = getattr(ds, "train_batch_sampler", None)
pp = importlib.import_module(cfg.post_process_pipeline)
post_process_pipeline, sweep_params, write_submission = pp.post_process_pipeline, pp.sweep_params, pp.write_submission
calc_metric = importlib.import_module(cfg.metric).calc_metric

# %%
# Run identity and tracking
git = git_state()
started = datetime.now(timezone.utc)
stamp = f"{started:%Y%m%d-%H%M%S}_{git['sha7']}{'-dirty' if git['dirty'] else ''}_seed{cfg.seed}"
run_dir = Path(cfg.output_dir) / stamp
run_dir.mkdir(parents=True, exist_ok=True)
run_id = f"{cfg.name}/{stamp}"

# W&B
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
IGNORED_DIRS = {".venv", "output", "wandb", "datamount", "input"}
run.log_code(root=".", exclude_fn=lambda path, _=None: bool(IGNORED_DIRS & set(Path(path).parts)))
run.define_metric("curr_step")
run.define_metric(name="*", step_metric="curr_step")

print(f"Run id:  {run_id}")
print(f"Run dir: {run_dir}")
print(f"wandb:   {run.url}")

# %%
# Data; OOF covers only cfg.folds, but each fold still trains on all the others
df = ds.load(cfg, "train")
oof_df = df[df["fold"].isin(cfg.folds)].reset_index(drop=True)

# Test loader, shared by every fold
test_df = ds.load(cfg, "test") if cfg.test else None
test_dataloader = None
if cfg.test:
    test_dataloader = DataLoader(
        CustomDataset(test_df, cfg=cfg, mode="test"),
        shuffle=False,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=val_collate_fn,
    )

# Decoder params for per-epoch validation
default_params = next(iter(sweep_params(cfg)))
print(f"{len(df)} train rows, OOF over folds {cfg.folds}: {len(oof_df)} rows")


def validate(eval_model, loader, val_df: pd.DataFrame, fold: int, label: str) -> tuple[dict, dict]:
    """Outputs and scores on the fold's validation rows, decoded with default params; logged."""
    val_data, val_loss = predict(eval_model, loader, batch_to_device, cfg.device, cfg.mixed_precision, cfg.amp_dtype)
    val_score = calc_metric(cfg, post_process_pipeline(cfg, val_data, val_df, **default_params))
    print(f"fold {fold} {label}: val_loss {val_loss:.4f}  " + "  ".join(f"{k} {v:.4f}" for k, v in val_score.items()))
    run.log(
        {
            "val/loss": val_loss,
            **{f"val/{k}": v for k, v in val_score.items()},
            "fold": fold,
            "curr_step": cfg.curr_step,
        }
    )
    return val_data, val_score


def track_best(best: dict, eval_model, val_data: dict, val_score: dict) -> None:
    """With cfg.keep_best, hold the best evaluation's outputs and a CPU copy of its weights."""
    score = val_score[cfg.selection_metric]
    if cfg.keep_best and score > best.get("score", -np.inf):
        state = {k: v.detach().to("cpu", copy=True) for k, v in eval_model.state_dict().items()}
        best.update(score=score, val_data=val_data, state=state, step=cfg.curr_step)


# Train each fold
oof_parts, test_sum, fold_seconds, best_evals = [], None, {}, {}
cfg.curr_step = 0
for fold in cfg.folds:
    # This fold's split and loaders
    print(f"\n----- fold {fold} -----")
    fold_t0 = time.time()
    set_seed(cfg.seed + fold)
    train_df = df[df["fold"] != fold]
    val_df = oof_df[oof_df["fold"] == fold]

    train_dataset = CustomDataset(train_df, cfg=cfg, mode="train")
    val_dataset = CustomDataset(val_df, cfg=cfg, mode="val")
    if train_batch_sampler is not None:
        batching = {"batch_sampler": train_batch_sampler(train_dataset, cfg)}
    else:
        batching = {"shuffle": True, "batch_size": cfg.batch_size}
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=tr_collate_fn,
        **batching,
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=val_collate_fn,
    )

    # %%
    # Model, optimizer, scheduler, scaler
    model = Net(dataset=train_dataset, cfg=cfg, mode="train").to(cfg.device)
    if hasattr(model, "optimizer_groups"):
        groups = model.optimizer_groups(cfg)
    else:
        groups = param_groups(model, cfg.weight_decay)
    optimizer = AdamW(groups, lr=cfg.lr)
    steps_per_epoch = len(train_dataset) // cfg.batch_size
    scheduler = SCHEDULES[cfg.schedule](
        optimizer,
        num_warmup_steps=int(cfg.warmup * steps_per_epoch),
        num_training_steps=cfg.epochs * steps_per_epoch,
    )
    # bf16 has fp32's range, so only fp16 needs its gradients scaled
    scaler = GradScaler(device=cfg.device, enabled=cfg.mixed_precision and cfg.amp_dtype == "float16")
    # use_buffers: BatchNorm running stats follow the averaged weights
    swa_model = None
    if cfg.swa_start_epoch is not None:
        swa_model = AveragedModel(model, multi_avg_fn=swa_average, use_buffers=True)
    eval_model = model
    best, last_score = {}, None

    # Training
    i = 0
    optimizer.zero_grad()
    for epoch in range(cfg.epochs):
        cfg.curr_epoch = epoch
        n_batches = len(train_dataloader)
        progress_bar = tqdm(range(n_batches), desc=f"fold {fold} train epoch {epoch}")
        tr_it = iter(train_dataloader)
        losses = []
        gc.collect()

        # One pass over the training batches
        model.train()
        torch.set_grad_enabled(True)
        for step, _ in enumerate(progress_bar, start=1):
            last = epoch == cfg.epochs - 1 and step == n_batches
            i += 1
            batch = batch_to_device(next(tr_it), cfg.device)
            cfg.curr_step += len(batch["idx"])
            total_grad_norm = None
            total_grad_norm_after_clip = None

            # Forward
            with autocast(cfg.device, enabled=cfg.mixed_precision, dtype=getattr(torch, cfg.amp_dtype)):
                output = model(batch)
            loss = output["loss"]
            losses.append(loss.item())
            progress_bar.set_postfix(loss=np.mean(losses[-10:]))

            # Backward; step every grad_accumulation batches, clipping if set
            scaler.scale(loss / cfg.grad_accumulation).backward()

            if i % cfg.grad_accumulation == 0:
                if cfg.track_grad_norm or cfg.clip_grad > 0:
                    scaler.unscale_(optimizer)
                if cfg.track_grad_norm:
                    total_grad_norm = calc_grad_norm(model.parameters())
                if cfg.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                    if cfg.track_grad_norm:
                        total_grad_norm_after_clip = calc_grad_norm(model.parameters())
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            scheduler.step()

            # Log
            log_dict = {f"train/{key}": output[key].item() for key in output if "loss" in key}
            log_dict["lr"] = optimizer.param_groups[0]["lr"]
            if total_grad_norm is not None:
                log_dict["total_grad_norm"] = total_grad_norm.item()
            if total_grad_norm_after_clip is not None:
                log_dict["total_grad_norm_after_clip"] = total_grad_norm_after_clip.item()
            run.log({**log_dict, "fold": fold, "curr_step": cfg.curr_step})

            # SWA every swa_steps batches, and after the last one
            swa_due = cfg.swa_steps and (step % cfg.swa_steps == 0 or last)
            if swa_model is not None and epoch >= cfg.swa_start_epoch and swa_due:
                swa_model.update_parameters(model)
                eval_model = swa_model.module

            # Validation every eval_steps batches, and after the last one
            if cfg.eval_steps and (step % cfg.eval_steps == 0 or last):
                val_data, last_score = validate(eval_model, val_dataloader, val_df, fold, f"step {cfg.curr_step}")
                track_best(best, eval_model, val_data, last_score)
                model.train()

        print(f"fold {fold} epoch {epoch}: train_loss {np.mean(losses):.4f}")

        # SWA: add this epoch's weights to the average
        if swa_model is not None and epoch >= cfg.swa_start_epoch and not cfg.swa_steps:
            swa_model.update_parameters(model)
            eval_model = swa_model.module

        # Validation with default decoder params; the full grid is swept on OOF
        if not cfg.eval_steps and ((epoch + 1) % cfg.eval_epochs == 0 or (epoch + 1) == cfg.epochs):
            val_data, last_score = validate(eval_model, val_dataloader, val_df, fold, f"epoch {epoch}")
            track_best(best, eval_model, val_data, last_score)

    # The best evaluation's outputs and weights
    if cfg.keep_best:
        val_data = best["val_data"]
        eval_model.load_state_dict(best["state"])
        best_evals[fold] = {
            "step": best["step"],
            "score": best["score"],
            "last_score": last_score[cfg.selection_metric],
        }
        print(
            f"fold {fold}: best at step {best['step']} ({best['score']:.4f}); last {best_evals[fold]['last_score']:.4f}"
        )

    # Fold outputs, from the SWA model when enabled
    oof_parts.append((val_df.index.to_numpy(), val_data))
    if test_dataloader is not None:
        test_data, _ = predict(
            eval_model, test_dataloader, batch_to_device, cfg.device, cfg.mixed_precision, cfg.amp_dtype
        )
        if test_sum is None:
            test_sum = test_data
        else:
            for k in test_sum:
                if k != "idx":
                    test_sum[k] += test_data[k]

    # Checkpoint
    if cfg.save_checkpoint:
        torch.save({"model": eval_model.state_dict()}, run_dir / f"fold{fold}.pth")
    fold_seconds[fold] = round(time.time() - fold_t0, 1)

    # Free GPU memory
    del model, eval_model, swa_model, optimizer, scheduler, scaler, train_dataloader, val_dataloader
    gc.collect()
    torch.cuda.empty_cache()

fit_seconds = round(sum(fold_seconds.values()), 1)
print("--- Training complete ---")

# %%
# OOF validation: sweep the decoder grid
oof_data = scatter(oof_parts, len(oof_df))
grid = list(sweep_params(cfg))
results = [{**params, **calc_metric(cfg, post_process_pipeline(cfg, oof_data, oof_df, **params))} for params in grid]
results_df = pd.DataFrame(results).sort_values(cfg.selection_metric, ascending=False)
best_params = {k: to_python(results_df.iloc[0][k]) for k in grid[0]}
print(results_df.to_string(index=False))

# Best params: overall and per-fold scores
best_pred = post_process_pipeline(cfg, oof_data, oof_df, **best_params)
cv = calc_metric(cfg, best_pred)
fold_scores = {int(f): calc_metric(cfg, g)[cfg.selection_metric] for f, g in best_pred.groupby("fold")}
print(f"\nBest OOF {cfg.selection_metric}: {cv[cfg.selection_metric]:.4f}  {best_params}")
print("per fold: " + "  ".join(f"{f}: {s:.4f}" for f, s in fold_scores.items()))


# %%
# Artifacts
def with_ids(frame: pd.DataFrame, data: dict) -> dict:
    return {id_col: frame[id_col].tolist(), **data} if id_col else data


# OOF
best_pred.to_csv(run_dir / "oof.csv", index=False)
results_df.to_csv(run_dir / "sweep.csv", index=False)
torch.save(with_ids(oof_df, oof_data), run_dir / "oof_outputs.pt")

# Test outputs, averaged over folds, and the submission
if test_sum is not None:
    n_folds = len(cfg.folds)
    test_data = {k: (v if k == "idx" else v / n_folds) for k, v in test_sum.items()}
    torch.save(with_ids(test_df, test_data), run_dir / "test_outputs.pt")
    predicted = post_process_pipeline(cfg, test_data, test_df, **best_params)
    submission = write_submission(cfg, predicted, run_dir / "submission.csv")
    shutil.copy2(run_dir / "submission.csv", "submission.csv")
    print(f"wrote {run_dir}/submission.csv ({len(submission)} rows)")

# %%
# Record the run
run.summary.update(
    {
        **cv,
        **{f"best_{k}": v for k, v in best_params.items()},
        **{f"fold{f}_{cfg.selection_metric}": s for f, s in fold_scores.items()},
        "fit_seconds": fit_seconds,
        "items": len(oof_df),
    }
)
run.log({"sweep": wandb.Table(dataframe=results_df)})

# run.json
manifest = {
    "run_id": run_id,
    "config": args.config,
    "overrides": overrides,
    "notes": cfg.notes,
    "wandb": {"id": run.id, "url": run.url, "project": cfg.wandb_project},
    "best_params": best_params,
    "cv": cv,
    "fold_scores": fold_scores,
    **({"best_evals": best_evals} if cfg.keep_best else {}),
    "folds": cfg.folds,
    "fit_seconds": fit_seconds,
    "fold_seconds": fold_seconds,
}
with open(run_dir / "run.json", "w") as f:
    json.dump(manifest, f, indent=2, default=str)

run.finish()

# Reproduce line
print(f"\n{run_id}  {cfg.selection_metric}={cv[cfg.selection_metric]:.4f}")
replay = [f"--{k} {shlex.quote(format_override(v))}" for k, v in overrides.items() if k != "seed"]
print(f"reproduce: python {Path(__file__).name} -C {args.config} --seed {cfg.seed} {' '.join(replay)}".rstrip())
print(f"           (from {git['describe']})")
