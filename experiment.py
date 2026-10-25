"""What the runners need beyond W&B: config overrides and key checks, and the local run identity."""

import importlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path


def git_state(required: bool = True) -> dict:
    """Commit and dirty flag of the tree; "nogit" outside a repository when not `required`.

    wandb records the commit but no dirty flag, and the run directory is named before wandb starts.
    """

    def git(*args):
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()

    # No repository: "nogit", when allowed
    if not required:
        try:
            git("rev-parse", "HEAD")
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {"sha": "nogit", "sha7": "nogit", "dirty": False, "describe": "nogit"}

    # Commit, short sha, dirty flag
    return {
        "sha": git("rev-parse", "HEAD"),
        "sha7": git("rev-parse", "--short=7", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "describe": git("describe", "--always", "--dirty"),
    }


def load_run_config(run_dir):
    """The config a train_nn.py run ran with: its config module plus the overrides in its run.json."""
    manifest = json.loads((Path(run_dir) / "run.json").read_text())
    cfg = deepcopy(importlib.import_module(f"configs.{manifest['config']}").cfg)
    for key, value in manifest.get("overrides", {}).items():
        setattr(cfg, key, value)
    return cfg


def coerce(raw: str, current):
    """Read an override string as the type the config already holds at that key."""
    if isinstance(current, bool):
        if raw not in ("True", "False"):
            raise SystemExit(f"expected True or False, got {raw!r}")
        return raw == "True"
    if isinstance(current, (list, dict)):
        return json.loads(raw)
    if current is None:
        # No type to copy: JSON when it parses, else a string
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return type(current)(raw)


def format_override(value) -> str:
    """Render an applied override so coerce() reads it back (bools via str: JSON's "false" is rejected)."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value)
    return json.dumps(value)


def to_python(value):
    """Unwrap a numpy scalar, which json.dump would write as a string (np.bool_ -> "True")."""
    return value.item() if hasattr(value, "item") else value


def check_config_keys(cfg, defaults) -> None:
    """Refuse a config key that neither default_config nor a configured module's CFG_KEYS declares."""
    known = set(vars(defaults))
    for path in (cfg.dataset, cfg.post_process_pipeline, cfg.metric, cfg.model):
        if path:
            known |= getattr(importlib.import_module(path), "CFG_KEYS", set())
    unknown = sorted(set(vars(cfg)) - known)
    if unknown:
        raise SystemExit(
            f"unknown config key(s) {unknown} — a typo, or declare them in "
            "configs/default_config.py or the owning module's CFG_KEYS"
        )


def apply_overrides(cfg, tokens: list[str]) -> dict:
    """Apply leftover `--key value` pairs onto cfg; an unknown key is an error."""
    if len(tokens) % 2:
        raise SystemExit(f"overrides must be --key value pairs, got {tokens}")
    applied = {}
    for flag, raw in zip(tokens[::2], tokens[1::2]):
        if not flag.startswith("--"):
            raise SystemExit(f"expected an option, got {flag!r}")
        key = flag.removeprefix("--").replace("-", "_")
        if key not in vars(cfg):
            raise SystemExit(f"unknown config key {key!r} (cfg has no such attribute)")
        value = coerce(raw, getattr(cfg, key))
        print(f"override cfg.{key}: {getattr(cfg, key)!r} -> {value!r}")
        setattr(cfg, key, value)
        applied[key] = value
    return applied
