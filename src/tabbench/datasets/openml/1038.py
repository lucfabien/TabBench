from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-1038"
    task: str  = "classification"
    target: str = "label"
    openml_id: int = 1038
    openml_name = "gina_agnostic"
