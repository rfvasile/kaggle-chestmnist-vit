"""CLI override parsing and validation, which W&B would record just as happily when wrong."""

from copy import deepcopy
from importlib import import_module

import pytest

from experiment import apply_overrides, coerce, format_override


@pytest.fixture
def cfg():
    return deepcopy(import_module("configs.cfg_b1").cfg)


# Overrides
def test_override_keeps_the_type_already_in_the_config(cfg):
    applied = apply_overrides(cfg, ["--seed", "7", "--n_folds", "3"])
    assert cfg.seed == 7 and isinstance(cfg.seed, int)
    assert cfg.n_folds == 3 and isinstance(cfg.n_folds, int)
    assert applied == {"seed": 7, "n_folds": 3}


def test_override_reads_bools_as_bools(cfg):
    """`--keep_predictor False` must not become the truthy string "False"."""
    apply_overrides(cfg, ["--keep_predictor", "False"])
    assert cfg.keep_predictor is False


def test_override_rejects_a_non_boolean_for_a_bool_key(cfg):
    with pytest.raises(SystemExit, match="expected True or False"):
        apply_overrides(cfg, ["--keep_predictor", "yes"])


def test_override_parses_json_for_a_none_valued_key(cfg):
    apply_overrides(cfg, ["--time_limit", "600", "--hyperparameters", '{"CAT": [{}]}'])
    assert cfg.time_limit == 600
    assert cfg.hyperparameters == {"CAT": [{}]}


def test_override_falls_back_to_a_string_when_a_none_key_is_not_json(cfg):
    cfg.presets = None
    apply_overrides(cfg, ["--presets", "best_quality"])
    assert cfg.presets == "best_quality"


def test_override_parses_json_for_a_list_key(cfg):
    apply_overrides(cfg, ["--folds", "[0, 1]"])
    assert cfg.folds == [0, 1]


def test_unknown_key_is_refused_rather_than_ignored(cfg):
    with pytest.raises(SystemExit, match="unknown config key 'treshold'"):
        apply_overrides(cfg, ["--treshold", "0.4"])


def test_dangling_override_is_refused(cfg):
    with pytest.raises(SystemExit, match="must be --key value pairs"):
        apply_overrides(cfg, ["--seed"])


def test_bare_value_without_a_flag_is_refused(cfg):
    with pytest.raises(SystemExit, match="expected an option"):
        apply_overrides(cfg, ["seed", "7"])


def test_empty_overrides_change_nothing(cfg):
    before = vars(cfg).copy()
    assert apply_overrides(cfg, []) == {}
    assert vars(cfg) == before


# coerce and the reproduce line
def test_coerce_leaves_an_existing_string_alone():
    """A str key must not go through JSON, or `--presets 600` becomes an int."""
    assert coerce("600", "medium") == "600"


@pytest.mark.parametrize("value", [7, 0.55, True, False, "medium", [0.4, 0.5], {"CAT": [{}]}])
def test_printed_reproduce_line_round_trips_through_coerce(value):
    assert coerce(format_override(value), value) == value
