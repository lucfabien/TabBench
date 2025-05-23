from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44992"
    task: str  = "regression"
    target: str = "FPS"
    openml_id: int = 44992
    openml_name = "fps_benchmark"
