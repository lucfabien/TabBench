from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44970"
    task: str  = "regression"
    target: str = "LC50"
    openml_id: int = 44970
    openml_name = "QSAR_fish_toxicity"
