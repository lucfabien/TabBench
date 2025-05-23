from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-488"
    task: str  = "classification"
    target: str = "Type"
    openml_id: int = 488
    openml_name = "colleges_aaup"
