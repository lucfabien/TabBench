from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44978"
    task: str  = "regression"
    target: str = "usr"
    openml_id: int = 44978
    openml_name = "cpu_activity"
