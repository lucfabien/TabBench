from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-469"
    task: str  = "classification"
    target: str = "Prevention"
    openml_id: int = 469
    openml_name = "analcatdata_dmft"
