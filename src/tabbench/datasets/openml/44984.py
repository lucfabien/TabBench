from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44984"
    task: str  = "regression"
    target: str = "wage"
    openml_id: int = 44984
    openml_name = "cps88wages"
