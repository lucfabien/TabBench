from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44957"
    task: str  = "regression"
    target: str = "sound_pressure"
    openml_id: int = 44957
    openml_name = "airfoil_self_noise"
