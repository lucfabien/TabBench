from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-6332"
    task: str  = "classification"
    target: str = "band_type"
    openml_id: int = 6332
    openml_name = "cylinder-bands"
