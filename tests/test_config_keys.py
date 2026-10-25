"""Config-file key validation; test_config_overrides covers the CLI."""

from copy import deepcopy
from importlib import import_module
from pathlib import Path

import pytest

from configs.default_config import cfg as defaults
from experiment import check_config_keys

CONFIGS = sorted(p.stem for p in Path(__file__).parents[1].glob("configs/cfg_*.py"))


@pytest.mark.parametrize("name", CONFIGS)
def test_shipped_config_declares_every_key_it_sets(name):
    check_config_keys(deepcopy(import_module(f"configs.{name}").cfg), defaults)


def test_misspelled_key_is_refused_rather_than_ignored():
    cfg = deepcopy(import_module("configs.cfg_b1").cfg)
    cfg.presetz = "extreme"
    with pytest.raises(SystemExit, match="presetz"):
        check_config_keys(cfg, defaults)


def test_a_module_key_is_known_only_while_its_module_is_the_one_configured():
    cfg = deepcopy(import_module("configs.cfg_t0").cfg)
    cfg.model = "models"
    with pytest.raises(SystemExit, match="dropout"):
        check_config_keys(cfg, defaults)
