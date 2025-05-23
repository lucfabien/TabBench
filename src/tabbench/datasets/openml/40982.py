from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-40982"
    task: str  = "classification"
    target: str = "target"
    openml_id: int = 40982
    openml_name = "steel-plates-fault"
