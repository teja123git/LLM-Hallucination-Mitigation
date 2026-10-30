"""YAML configuration with attribute access, profiles and CLI overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


class Cfg(dict):
    """A dict whose keys are also attributes (recursively)."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - trivial
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Cfg({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_override(item: str) -> Dict[str, Any]:
    if "=" not in item:
        raise ValueError(f"Override must look like key.sub=value, got: {item!r}")
    key, raw = item.split("=", 1)
    value = yaml.safe_load(raw)
    node: Dict[str, Any] = {}
    cur = node
    parts = key.strip().split(".")
    for p in parts[:-1]:
        cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value
    return node


def load_config(
    path: Optional[str | Path] = None,
    overrides: Optional[Iterable[str]] = None,
    mock: Optional[bool] = None,
) -> Cfg:
    """Load ``configs/default.yaml`` and merge an optional profile file and ``a.b=c`` overrides."""
    with open(CONFIG_DIR / "default.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if path:
        p = Path(path)
        if not p.exists() and (CONFIG_DIR / p).exists():
            p = CONFIG_DIR / p
        with open(p, "r", encoding="utf-8") as f:
            data = deep_update(data, yaml.safe_load(f) or {})
    if mock:   # mock settings always win over a profile, so a mock run can never reach the real API
        with open(CONFIG_DIR / "mock.yaml", "r", encoding="utf-8") as f:
            data = deep_update(data, yaml.safe_load(f) or {})
    for item in overrides or []:
        data = deep_update(data, _parse_override(item))
    cfg = _wrap(data)
    # resolve directories relative to the repository root
    for key in ("data_dir", "results_dir"):
        p = Path(cfg.project[key])
        cfg.project[key] = str(p if p.is_absolute() else ROOT / p)
    return cfg


def results_dir(cfg: Cfg, stage: str | None = None) -> Path:
    p = Path(cfg.project.results_dir)
    if stage:
        p = p / stage
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_dir(cfg: Cfg, *parts: str) -> Path:
    p = Path(cfg.project.data_dir).joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
