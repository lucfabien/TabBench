from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44971"
    task: str  = "classification"
    target: str = "quality"
    openml_id: int = 44971
    openml_name = "white_wine"
