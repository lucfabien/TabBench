from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44994"
    task: str  = "regression"
    target: str = "Price"
    openml_id: int = 44994
    openml_name = "cars"
