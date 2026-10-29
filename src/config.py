"""Configuration loading and validation (plain JSON, no extra libraries)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default_config.json"

_REQUIRED_NODE_KEYS = {"node_id", "site_id", "tier", "cpu_cores", "mem_gb",
                       "idle_power_w", "max_power_w", "pue", "latency_ms"}


def resolve_path(path_str: str | Path) -> Path:
    """Resolve a path relative to the project root (never hard-coded absolute)."""
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def validate_config(cfg: dict) -> None:
    """Raise ConfigError with a readable message if the config is inconsistent."""
    for key in ("data", "forecaster", "sites", "nodes", "migration",
                "carbon_aware_scheduler", "synthetic_workload"):
        if key not in cfg:
            raise ConfigError(f"Config is missing the '{key}' section.")
    if cfg["time_step_minutes"] <= 0:
        raise ConfigError("time_step_minutes must be positive.")
    if cfg["data"]["mode"] not in ("sample", "real"):
        raise ConfigError("data.mode must be 'sample' or 'real'.")
    site_ids = {s["site_id"] for s in cfg["sites"]}
    if len(site_ids) != len(cfg["sites"]):
        raise ConfigError("Duplicate site_id in 'sites'.")
    if not cfg["nodes"]:
        raise ConfigError("At least one node is required.")
    seen = set()
    for n in cfg["nodes"]:
        missing = _REQUIRED_NODE_KEYS - set(n)
        if missing:
            raise ConfigError(f"Node {n.get('node_id')} is missing keys: {sorted(missing)}")
        if n["node_id"] in seen:
            raise ConfigError(f"Duplicate node_id {n['node_id']}")
        seen.add(n["node_id"])
        if n["site_id"] not in site_ids:
            raise ConfigError(f"Node {n['node_id']} refers to unknown site '{n['site_id']}'.")
        if n["tier"] not in ("edge", "cloud"):
            raise ConfigError(f"Node {n['node_id']}: tier must be 'edge' or 'cloud'.")
        if not (0 <= n["idle_power_w"] <= n["max_power_w"]):
            raise ConfigError(f"Node {n['node_id']}: need 0 <= idle_power_w <= max_power_w.")
        if n["cpu_cores"] <= 0 or n["mem_gb"] <= 0 or n["pue"] < 1.0:
            raise ConfigError(f"Node {n['node_id']}: cpu_cores, mem_gb must be >0 and pue >= 1.")
    if cfg["forecaster"]["max_horizon_steps"] < 2:
        raise ConfigError("forecaster.max_horizon_steps must be >= 2.")
    if cfg["data"]["simulation_days"] <= 0:
        raise ConfigError("data.simulation_days must be positive.")
    if cfg["carbon_aware_scheduler"].get("max_deferral_steps", 12) < 0:
        raise ConfigError("carbon_aware_scheduler.max_deferral_steps must be >= 0.")
    if not 0 <= cfg["carbon_aware_scheduler"]["defer_margin"] < 1:
        raise ConfigError("carbon_aware_scheduler.defer_margin must be in [0, 1).")
    if cfg["migration"]["max_migrations_per_job"] < 0:
        raise ConfigError("migration.max_migrations_per_job must be >= 0.")
    if cfg["migration"]["bandwidth_mbps"] <= 0 or cfg["migration"]["transfer_kwh_per_gb"] < 0:
        raise ConfigError("migration bandwidth must be > 0 and transfer energy must be >= 0.")


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load default config, optionally merge a user JSON file and CLI overrides."""
    try:
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"Default config not found at {DEFAULT_CONFIG_PATH}") from exc
    if path:
        p = resolve_path(path)
        try:
            with open(p, "r", encoding="utf-8") as fh:
                user_cfg = json.load(fh)
        except FileNotFoundError as exc:
            raise ConfigError(f"Config file not found: {p}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Config file {p} is not valid JSON: {exc}") from exc
        _deep_update(cfg, user_cfg)
    if overrides:
        _deep_update(cfg, copy.deepcopy(overrides))
    validate_config(cfg)
    return cfg
