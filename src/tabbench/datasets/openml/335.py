from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-335"
    task: str  = "classification"
    target: str = "class"
    openml_id: int = 335
    openml_name = "monks-problems-3"
