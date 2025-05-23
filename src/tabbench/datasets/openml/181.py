from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-181"
    task: str  = "classification"
    target: str = "class_protein_localization"
    openml_id: int = 181
    openml_name = "yeast"
