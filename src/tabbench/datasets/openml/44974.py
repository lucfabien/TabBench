from dataclasses import dataclass

from neuralk_foundry_ce.datasets.base import OpenMLDataConfig


@dataclass
class DataConfig(OpenMLDataConfig):
    name: str  = "openml-44974"
    task: str  = "regression"
    target: str = "utime"
    openml_id: int = 44974
    openml_name = "video_transcoding"
