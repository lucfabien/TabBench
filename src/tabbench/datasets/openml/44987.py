from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44987"
    task: str  = "regression"
    target: str = "counts_for_sons_current_occupation"
    openml_id: int = 44987
    openml_name = "socmob"
