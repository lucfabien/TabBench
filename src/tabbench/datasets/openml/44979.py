from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44979"
    task: str  = "regression"
    target: str = "price"
    openml_id: int = 44979
    openml_name = "diamonds"
