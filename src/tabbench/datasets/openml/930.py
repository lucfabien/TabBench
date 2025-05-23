from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-930"
    task: str  = "classification"
    target: str = "binaryClass"
    openml_id: int = 930
    openml_name = "colleges_usnews"
