from collections import OrderedDict
from pathlib import Path

from neuralk_foundry_ce.datasets import LoadDataset
from neuralk_foundry_ce.workflow.use_cases import Classification as _Classification
from neuralk_foundry_ce.workflow.use_cases import Categorisation as _Categorisation


class Classification(_Classification):

    def __init__(self, dataset_name, cache_dir=None):
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
        super().__init__(cache_dir=cache_dir)
        self.steps['Dataset'] = LoadDataset(dataset=dataset_name)
        # Set the dataset loading as first step
        self.steps.move_to_end('Dataset', last=False)

    def run(self, fold_index=0):
        return self._run({'fold_index': fold_index})
    
    def notebook_display(self, level=1):
        return super().notebook_display(level)



class Categorisation(_Categorisation):

    def __init__(self, dataset_name, cache_dir=None):
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
        super().__init__(cache_dir=cache_dir)
        self.steps['Dataset'] = LoadDataset(dataset=dataset_name)
        # Set the dataset loading as first step
        self.steps.move_to_end('Dataset', last=False)

    def run(self, fold_index=0):
        return self._run({'fold_index': fold_index})

    def notebook_display(self, level=1):
        return super().notebook_display(level)
