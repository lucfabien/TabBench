from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-188"
    task: str  = "classification"
    target: str = "Utility"
    openml_id: int = 188
    openml_name = "eucalyptus"
