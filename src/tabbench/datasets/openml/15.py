from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-15"
    task: str  = "classification"
    target: str = "Class"
    openml_id: int = 15
    openml_name = "breast-w"
