from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44977"
    task: str  = "regression"
    target: str = "medianHouseValue"
    openml_id: int = 44977
    openml_name = "california_housing"
