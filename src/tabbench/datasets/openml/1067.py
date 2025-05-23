from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-1067"
    task: str  = "classification"
    target: str = "defects"
    openml_id: int = 1067
    openml_name = "kc1"
