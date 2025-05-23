from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44960"
    task: str  = "regression"
    target: str = "heating_load"
    openml_id: int = 44960
    openml_name = "energy_efficiency"
