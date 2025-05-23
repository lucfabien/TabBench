from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-50"
    task: str  = "classification"
    target: str = "Class"
    openml_id: int = 50
    openml_name = "tic-tac-toe"
