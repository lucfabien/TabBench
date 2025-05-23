from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44956"
    task: str  = "regression"
    target: str = "rings"
    openml_id: int = 44956
    openml_name = "abalone"
