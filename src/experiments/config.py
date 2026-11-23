"""Experiment configuration: TOML files, environment-variable paths, overrides.

Paths may use ${VAR} or ${VAR:-default}, so no username, scratch path or
cluster detail is written into a config. Secrets are never read from config:
API keys come from the environment only, and keys whose name looks like a
secret are rejected.

    cfg = load_config("configs/experiments/mtg_ad_dev.toml", ["generation.max_new_tokens=96"])
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tomllib
from typing import Any

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_SECRET_KEY = re.compile(r"(?i)(api_?key|secret|token|password)$")

REQUIRED = ("experiment.name", "dataset.name", "paths.cache_root", "paths.output_root")

DEFAULTS: dict[str, Any] = {
    "experiment": {"seed": 0, "allow_test": False},
    "selection": {
        "tolerance_seconds": 2.0,
        "duration_slack_seconds": 1.0,
        "num_frames": 4,
        "span_seconds": 2.0,
        "bundle": {},
    },
    "generation": {
        "max_new_tokens": 64,
        "temperature": 0.0,
        "word_limit_policy": "truncate",
        "timeline": False,
        "timeline_window_seconds": 30.0,
    },
    "evaluation": {"resamples": 10000, "tts_words_per_second": 2.5, "tts_speed": 1.2},
    "matrix": {"selectors": ["U", "V", "A", "AV", "O"], "priors": ["P0", "P1"], "models": []},
    "ablations": {"enabled": []},
}


class ConfigError(ValueError):
    pass


def expand_env(value: Any, env: dict[str, str] | None = None) -> Any:
    env = os.environ if env is None else env
    if isinstance(value, str):
        def sub(match):
            name, default = match.group(1), match.group(2)
            if name in env and env[name] != "":
                return env[name]
            if default is not None:
                return default
            raise ConfigError(f"environment variable {name} is not set and has no default")
        return _ENV.sub(sub, value)
    if isinstance(value, list):
        return [expand_env(v, env) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v, env) for k, v in value.items()}
    return value


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _parse_scalar(text: str) -> Any:
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Dotted-key overrides: 'generation.max_new_tokens=96', 'matrix.priors=["P0"]'."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"override must be key=value: {item!r}")
        key, text = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"cannot set {key}: {part} is not a table")
        node[parts[-1]] = _parse_scalar(text.strip())
    return cfg


def _check_no_secrets(node: Any, path: str = "") -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if _SECRET_KEY.search(k) and not k.endswith("_env"):
                raise ConfigError(f"config key {path + k!r} looks like a secret; use an environment variable (e.g. {k}_env)")
            _check_no_secrets(v, f"{path}{k}.")
    elif isinstance(node, list):
        for v in node:
            _check_no_secrets(v, path)


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def validate(cfg: dict) -> dict:
    for key in REQUIRED:
        if get(cfg, key) in (None, ""):
            raise ConfigError(f"missing required config key {key}")
    if float(get(cfg, "generation.temperature", 0.0)) != 0.0:
        raise ConfigError("generation.temperature is a held-constant control and must be 0.0")
    if get(cfg, "dataset.split") == "test" and not get(cfg, "experiment.allow_test"):
        raise ConfigError("dataset.split = 'test' requires experiment.allow_test = true (report on test once, never tune on it)")
    _check_no_secrets(cfg)
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None, env: dict[str, str] | None = None) -> dict:
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)
    cfg = deep_merge(DEFAULTS, raw)
    cfg = apply_overrides(cfg, overrides)
    cfg = expand_env(cfg, env)
    cfg["_source"] = str(path)
    return validate(cfg)


def stable_hash(obj: Any, length: int = 10) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:length]
