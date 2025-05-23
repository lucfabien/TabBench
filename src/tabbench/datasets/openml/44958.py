from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44958"
    task: str  = "regression"
    target: str = "verification.time"
    openml_id: int = 44958
    openml_name = "auction_verification"
