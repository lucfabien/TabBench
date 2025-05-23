from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44976"
    task: str  = "regression"
    target: str = "V22"
    openml_id: int = 44976
    openml_name = "sarcos"
