from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44962"
    task: str  = "regression"
    target: str = "area"
    openml_id: int = 44962
    openml_name = "forest_fires"
