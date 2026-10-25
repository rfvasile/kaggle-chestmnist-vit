"""Seed spread per config: the aggregation the W&B UI does not do.

python scripts/runs.py                  # every config in the project
python scripts/runs.py cfg_b1           # one config
"""

import sys
from importlib import import_module

import pandas as pd

import wandb

PROJECT = import_module("configs.default_config").cfg.wandb_project


def seed_spread(project: str = PROJECT, config_name: str | None = None) -> tuple[pd.DataFrame, str]:
    """One row per config: seed count and spread of the runs' most common selection metric."""
    # Finished runs; lazy=False because lazily listed runs have an empty config, which load() does not fill
    filters = {"config.name": config_name} if config_name else None
    runs = [r for r in wandb.Api().runs(project, filters=filters, lazy=False) if r.state == "finished"]
    if not runs:
        raise SystemExit(f"no finished runs in {project}" + (f" for {config_name}" if config_name else ""))

    # The metric most runs are ranked by
    rows = [{"config": r.config.get("name"), "metric": r.config.get("selection_metric"), **r.summary} for r in runs]
    frame = pd.DataFrame(rows)
    declared = frame["metric"].dropna()
    if declared.empty:
        raise SystemExit(f"{len(runs)} runs carry no cfg.selection_metric; cannot rank them")
    metric = str(declared.mode().iloc[0])

    # Spread per config
    return (
        frame.groupby("config")[metric]
        .agg(n="count", mean="mean", std="std", best="max", worst="min")
        .sort_values("mean", ascending=False)
        .round(6)
    ), metric


if __name__ == "__main__":
    summary, metric = seed_spread(config_name=sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"{metric} across seeds:")
    print(summary.to_string())
