from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44967"
    task: str  = "classification"
    target: str = "G3"
    openml_id: int = 44967
    openml_name = "student_performance_por"
