from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class RapidUnConfig:
    """
    Thin wrapper around a YAML configuration dictionary for RapidUn.

    The YAML file is expected to have (at least) the following top-level keys:
      - model
      - data
      - training
      - experiment (optional)
    """
    cfg: Dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str) -> "RapidUnConfig":
        """
        Load a RapidUnConfig from a YAML file.
        """
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(cfg)

    def __getitem__(self, key: str) -> Any:
        """
        Allow dict-style access: config["model"], config["data"], etc.
        """
        return self.cfg[key]

    def get(self, key: str, default=None) -> Any:
        """
        Safe dictionary-style access with a default value.
        """
        return self.cfg.get(key, default)

    @property
    def model(self) -> Dict[str, Any]:
        """
        Convenience accessor for the 'model' section of the config.
        """
        return self.cfg["model"]

    @property
    def data(self) -> Dict[str, Any]:
        """
        Convenience accessor for the 'data' section of the config.
        """
        return self.cfg["data"]

    @property
    def training(self) -> Dict[str, Any]:
        """
        Convenience accessor for the 'training' section of the config.
        """
        return self.cfg["training"]

    @property
    def experiment(self) -> Dict[str, Any]:
        """
        Convenience accessor for the optional 'experiment' section.
        Returns an empty dict if not present.
        """
        return self.cfg.get("experiment", {})
