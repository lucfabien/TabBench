from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44972"
    task: str  = "classification"
    target: str = "quality"
    openml_id: int = 44972
    openml_name = "red_wine"
