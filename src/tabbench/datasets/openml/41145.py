from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-41145"
    task: str  = "classification"
    target: str = "class"
    openml_id: int = 41145
    openml_name = "philippine"
