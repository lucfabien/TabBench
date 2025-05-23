from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-179"
    task: str  = "classification"
    target: str = "class"
    openml_id: int = 179
    openml_name = "adult"
