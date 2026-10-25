"""Every config must satisfy its runner's contract before it is worth a fit."""

import inspect
from copy import deepcopy
from importlib import import_module
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CONFIG_NAMES = sorted(p.stem for p in (ROOT / "configs").glob("cfg_*.py"))

# runner -> seam -> names the runner imports from that module
SEAMS = {
    "train_gluon.py": {
        "dataset": ("build", "load"),
        "post_process_pipeline": ("post_process_pipeline", "sweep_params", "write_submission"),
        "metric": ("calc_metric",),
    },
    "train_nn.py": {
        "dataset": ("load", "CustomDataset", "tr_collate_fn", "val_collate_fn", "batch_to_device"),
        "model": ("Net",),
        "post_process_pipeline": ("post_process_pipeline", "sweep_params", "write_submission"),
        "metric": ("calc_metric",),
    },
}


# Helpers
def load_cfg(name):
    return deepcopy(import_module(f"configs.{name}").cfg)


def runner_of(cfg) -> str:
    return "train_nn.py" if cfg.model else "train_gluon.py"


# Config discovery
def test_configs_were_found():
    assert CONFIG_NAMES, "no configs/cfg_*.py found — the parametrised tests below would vacuously pass"


def test_every_experiment_config_uses_the_cfg_prefix():
    """CONFIG_NAMES globs cfg_*, so a config named otherwise would silently skip every test here."""
    EXEMPT = {"__init__", "default_config"}
    stray = sorted(
        p.stem for p in (ROOT / "configs").glob("*.py") if p.stem not in EXEMPT and not p.stem.startswith("cfg_")
    )
    assert not stray, f"configs/ holds {stray}, which no contract test will ever check; rename to cfg_*.py"


# Per-config contract
@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_config_name_matches_its_filename(name):
    assert load_cfg(name).name == name


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_config_seams_expose_what_the_runner_imports(name):
    cfg = load_cfg(name)
    for seam, required in SEAMS[runner_of(cfg)].items():
        module = import_module(getattr(cfg, seam))
        missing = [fn for fn in required if not callable(getattr(module, fn, None))]
        assert not missing, f"{name}: {getattr(cfg, seam)} is missing {missing}"


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_sweep_grid_is_non_empty_and_matches_the_postprocess_signature(name):
    cfg = load_cfg(name)
    pp = import_module(cfg.post_process_pipeline)
    grid = list(pp.sweep_params(cfg))
    assert grid, f"{name}: sweep_params yielded nothing"

    accepted = set(inspect.signature(pp.post_process_pipeline).parameters)
    for params in grid:
        assert set(params) <= accepted, f"{name}: {set(params) - accepted} not accepted by post_process_pipeline"


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_selection_metric_is_produced_by_the_metric_module(name):
    cfg = load_cfg(name)
    source = inspect.getsource(import_module(cfg.metric).calc_metric)
    assert f'"{cfg.selection_metric}"' in source, f"{name}: calc_metric never emits {cfg.selection_metric!r}"


# Runners
@pytest.mark.parametrize("runner", SEAMS)
def test_runner_reads_every_seam_declared_here(runner):
    """Keeps SEAMS in sync; read, not imported, since importing a runner starts a fit."""
    source = (ROOT / runner).read_text()
    for seam in SEAMS[runner]:
        assert f"cfg.{seam}" in source, f"{runner} no longer reads cfg.{seam}"
