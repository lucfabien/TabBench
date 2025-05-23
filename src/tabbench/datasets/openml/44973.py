from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44973"
    task: str  = "regression"
    target: str = "stab"
    openml_id: int = 44973
    openml_name = "grid_stability"
