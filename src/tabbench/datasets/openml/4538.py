from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-4538"
    task: str  = "classification"
    target: str = "Phase"
    openml_id: int = 4538
    openml_name = "GesturePhaseSegmentationProcessed"
