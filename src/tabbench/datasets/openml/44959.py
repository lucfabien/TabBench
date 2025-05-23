from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44959"
    task: str  = "regression"
    target: str = "strength"
    openml_id: int = 44959
    openml_name = "concrete_compressive_strength"
