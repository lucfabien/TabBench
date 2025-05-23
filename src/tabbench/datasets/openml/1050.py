from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-1050"
    task: str  = "classification"
    target: str = "c"
    openml_id: int = 1050
    openml_name = "pc3"
