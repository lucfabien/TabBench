from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-45402"
    task: str  = "regression"
    target: str = "ln_votes_pop"
    openml_id: int = 45402
    openml_name = "space_ga"
