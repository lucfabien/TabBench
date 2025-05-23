from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44975"
    task: str  = "regression"
    target: str = "energy_total"
    openml_id: int = 44975
    openml_name = "wave_energy"
