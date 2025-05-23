from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44993"
    task: str  = "regression"
    target: str = "whrswk"
    openml_id: int = 44993
    openml_name = "health_insurance"
