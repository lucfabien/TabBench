from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-458"
    task: str  = "classification"
    target: str = "Author"
    openml_id: int = 458
    openml_name = "analcatdata_authorship"
