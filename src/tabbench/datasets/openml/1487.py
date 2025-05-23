from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-1487"
    task: str  = "classification"
    target: str = "Class"
    openml_id: int = 1487
    openml_name = "ozone-level-8hr"
