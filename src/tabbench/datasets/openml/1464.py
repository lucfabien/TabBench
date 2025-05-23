from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-1464"
    task: str  = "classification"
    target: str = "Class"
    openml_id: int = 1464
    openml_name = "blood-transfusion-service-center"
