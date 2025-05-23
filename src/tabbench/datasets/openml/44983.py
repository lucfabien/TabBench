from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44983"
    task: str  = "regression"
    target: str = "SALE_PRC"
    openml_id: int = 44983
    openml_name = "miami_housing"
