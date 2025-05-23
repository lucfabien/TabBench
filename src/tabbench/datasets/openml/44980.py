from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44980"
    task: str  = "regression"
    target: str = "y"
    openml_id: int = 44980
    openml_name = "kin8nm"
