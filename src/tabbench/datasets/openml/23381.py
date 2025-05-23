from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-23381"
    task: str  = "classification"
    target: str = "Class"
    openml_id: int = 23381
    openml_name = "dresses-sales"
