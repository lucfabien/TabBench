from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44964"
    task: str  = "regression"
    target: str = "critical_temp"
    openml_id: int = 44964
    openml_name = "superconductivity"
