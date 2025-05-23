from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-1475"
    task: str  = "classification"
    target: str = "Class"
    openml_id: int = 1475
    openml_name = "first-order-theorem-proving"
