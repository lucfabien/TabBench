from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-40994"
    task: str  = "classification"
    target: str = "outcome"
    openml_id: int = 40994
    openml_name = "climate-model-simulation-crashes"
