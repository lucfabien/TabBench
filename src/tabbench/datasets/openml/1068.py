from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-1068"
    task: str  = "classification"
    target: str = "defects"
    openml_id: int = 1068
    openml_name = "pc1"
