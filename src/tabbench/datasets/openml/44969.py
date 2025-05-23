from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44969"
    task: str  = "regression"
    target: str = "gt_compressor_decay_state_coefficient"
    openml_id: int = 44969
    openml_name = "naval_propulsion_plant"
