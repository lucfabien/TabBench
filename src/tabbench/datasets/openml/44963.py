from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44963"
    task: str  = "regression"
    target: str = "RMSD"
    openml_id: int = 44963
    openml_name = "physiochemical_protein"
