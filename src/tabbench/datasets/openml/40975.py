from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-40975"
    task: str  = "classification"
    target: str = "class"
    openml_id: int = 40975
    openml_name = "car"
