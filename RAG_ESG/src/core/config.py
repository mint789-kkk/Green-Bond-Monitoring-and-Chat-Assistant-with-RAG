"""Configuration management for DeskRAG."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

CONFIG_DIR = Path(os.path.expanduser("~")) / ".deskrag"
CONFIG_FILE = CONFIG_DIR / "config.json"


@dataclass
class ModelRegistry:
    embeddings: List[str] = field(default_factory=list)
    llms: List[str] = field(default_factory=list)


@dataclass
class AppConfig:
    data_dir: Path = CONFIG_DIR / "data"
    cache_dir: Path = CONFIG_DIR / "cache"
    model_registry: ModelRegistry = field(default_factory=ModelRegistry)

    def to_dict(self) -> Dict:
        return {
            "data_dir": str(self.data_dir),
            "cache_dir": str(self.cache_dir),
            "model_registry": {
                "embeddings": self.model_registry.embeddings,
                "llms": self.model_registry.llms,
            },
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "AppConfig":
        registry = payload.get("model_registry", {})
        return cls(
            data_dir=Path(payload.get("data_dir", CONFIG_DIR / "data")),
            cache_dir=Path(payload.get("cache_dir", CONFIG_DIR / "cache")),
            model_registry=ModelRegistry(
                embeddings=registry.get("embeddings", []),
                llms=registry.get("llms", []),
            ),
        )


def ensure_dirs(config: AppConfig) -> None:
    for path in [CONFIG_DIR, config.data_dir, config.cache_dir]:
        path.mkdir(parents=True, exist_ok=True)


def load_config() -> AppConfig:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        config = AppConfig.from_dict(payload)
    else:
        config = AppConfig()
        save_config(config)
    ensure_dirs(config)
    return config


def save_config(config: AppConfig) -> None:
    ensure_dirs(config)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)



