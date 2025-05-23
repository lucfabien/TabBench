from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-40498"
    task: str  = "classification"
    target: str = "Class"
    openml_id: int = 40498
    openml_name = "wine-quality-white"
