from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44966"
    task: str  = "classification"
    target: str = "c_class_flares"
    openml_id: int = 44966
    openml_name = "solar_flare"
