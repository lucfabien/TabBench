from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-41021"
    task: str  = "regression"
    target: str = "RS"
    openml_id: int = 41021
    openml_name = "Moneyball"
