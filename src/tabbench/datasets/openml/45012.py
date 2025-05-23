from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-45012"
    task: str  = "regression"
    target: str = "wage_eur"
    openml_id: int = 45012
    openml_name = "fifa"
