"""What inference outside the repo relies on: no git, and a run rebuilt from run.json."""

import json
import os
import subprocess

import pytest

from experiment import git_state, load_run_config


def test_git_state_without_a_repository_is_nogit_only_when_allowed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", os.fspath(tmp_path.parent))
    assert git_state(required=False)["sha"] == "nogit"
    with pytest.raises(subprocess.CalledProcessError):
        git_state()


def test_load_run_config_applies_the_run_overrides(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"config": "cfg_t0", "overrides": {"epochs": 1, "lr": 1e-4}}))
    cfg = load_run_config(tmp_path)
    assert (cfg.name, cfg.epochs, cfg.lr) == ("cfg_t0", 1, 1e-4)
