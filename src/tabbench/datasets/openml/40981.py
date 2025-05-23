from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-40981"
    task: str  = "classification"
    target: str = "A15"
    openml_id: int = 40981
    openml_name = "Australian"
