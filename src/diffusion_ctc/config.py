"""Configuration loader.

All paths, HF repo IDs, and (optionally) API keys live in `config.yaml` at
the repo root. Anything left null in the YAML falls back to a same-named
environment variable; if both are absent, defaults from this module apply.

Typical usage:

    from diffusion_ctc.config import load_config
    cfg = load_config()
    model_id = cfg.models.llada
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "config.yaml"
_EXAMPLE_CONFIG = _REPO_ROOT / "config.example.yaml"


@dataclass
class ModelsCfg:
    llada: str = "GSAI-ML/LLaDA-8B-Instruct"
    llada_moe: str = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"
    llada_local: Optional[str] = None
    llada_moe_local: Optional[str] = None

    def resolve(self, variant: str) -> str:
        """Return a local path if set, else the HF repo id."""
        if variant == "llada":
            return self.llada_local or self.llada
        if variant == "moe":
            return self.llada_moe_local or self.llada_moe
        raise ValueError(f"Unknown model variant: {variant!r}")


@dataclass
class DatasetsCfg:
    source_magpie: str = "Magpie-Align/Magpie-Pro-300K-Filtered"
    sft_vanilla: str = "GAIR/lima"
    # Pre-tokenized Magpie-Pro [512, 1024] subsets produced by
    # scripts/preprocess_magpie{,_moe}.py:
    #   sft_baseline_*  - raw instruction (no slack-priming system prompt);
    #                     consumed by the CE-Only baseline.
    #   sft_slack_*     - slack-priming system prompt prepended to each
    #                     instruction; consumed by Jitter and CE+CTC-S.
    sft_baseline_llada: str = ""
    sft_baseline_moe: str = ""
    sft_slack_llada: str = ""
    sft_slack_moe: str = ""


@dataclass
class PathsCfg:
    output_dir: str = "./checkpoints"
    eval_outputs: str = "./eval_outputs"


@dataclass
class ApiKeysCfg:
    hf_token: Optional[str] = None
    hf_up_token: Optional[str] = None

    def resolve(self) -> "ApiKeysCfg":
        """Fall back to env vars (HF_TOKEN, HF_UP_TOKEN) for any unset key."""
        return ApiKeysCfg(
            hf_token=self.hf_token or os.environ.get("HF_TOKEN"),
            hf_up_token=self.hf_up_token or os.environ.get("HF_UP_TOKEN"),
        )


@dataclass
class Config:
    models: ModelsCfg = field(default_factory=ModelsCfg)
    datasets: DatasetsCfg = field(default_factory=DatasetsCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)
    api_keys: ApiKeysCfg = field(default_factory=ApiKeysCfg)


def _build_section(cls, raw):
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise TypeError(f"Expected mapping for {cls.__name__}, got {type(raw).__name__}")
    fields = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = set(raw) - fields
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**{k: raw[k] for k in raw})


def load_config(path: Optional[str | os.PathLike] = None) -> Config:
    """Load configuration from a YAML file.

    Resolution order for the YAML file:
      1. The `path` argument, if provided.
      2. The `DIFFUSION_CTC_CONFIG` environment variable.
      3. `config.yaml` at the repo root.
      4. `config.example.yaml` at the repo root (defaults only).
    """
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    env_path = os.environ.get("DIFFUSION_CTC_CONFIG")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([_DEFAULT_CONFIG, _EXAMPLE_CONFIG])

    yaml_path = next((p for p in candidates if p.exists()), None)
    if yaml_path is None:
        return Config(api_keys=ApiKeysCfg().resolve())

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = Config(
        models=_build_section(ModelsCfg, raw.get("models")),
        datasets=_build_section(DatasetsCfg, raw.get("datasets")),
        paths=_build_section(PathsCfg, raw.get("paths")),
        api_keys=_build_section(ApiKeysCfg, raw.get("api_keys")),
    )
    cfg.api_keys = cfg.api_keys.resolve()
    return cfg
