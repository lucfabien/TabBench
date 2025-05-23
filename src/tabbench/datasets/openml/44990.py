from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44990"
    task: str  = "regression"
    target: str = "total"
    openml_id: int = 44990
    openml_name = "brazilian_houses"
