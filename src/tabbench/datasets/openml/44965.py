from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44965"
    task: str  = "regression"
    target: str = "latitude"
    openml_id: int = 44965
    openml_name = "geographical_origin_of_music"
