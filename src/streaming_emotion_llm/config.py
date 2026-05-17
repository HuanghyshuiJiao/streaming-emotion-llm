from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    values: dict


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    return ExperimentConfig(path=config_path, values=values)
