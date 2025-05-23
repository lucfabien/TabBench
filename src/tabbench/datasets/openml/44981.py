from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44981"
    task: str  = "regression"
    target: str = "thetadd6"
    openml_id: int = 44981
    openml_name = "pumadyn32nh"
