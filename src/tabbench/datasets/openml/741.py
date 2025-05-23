from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-741"
    task: str  = "classification"
    target: str = "binaryClass"
    openml_id: int = 741
    openml_name = "rmftsa_sleepdata"
