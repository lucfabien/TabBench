from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-1063"
    task: str  = "classification"
    target: str = "problems"
    openml_id: int = 1063
    openml_name = "kc2"
