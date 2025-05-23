from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-40900"
    task: str  = "classification"
    target: str = "Target"
    openml_id: int = 40900
    openml_name = "Satellite"
