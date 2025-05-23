from pathlib import Path
import json
import shutil
import os
import pandas as pd
import glob


from sklearn.datasets import make_classification

from neuralk_foundry_ce.workflow.utils import make_json_serializable
from neuralk_foundry_ce.models.classifier import (
    LightGBMClassifier, XGBoostClassifier, CatBoostClassifier,
    MLPClassifier,
    TabICLClassifier, TabPFNClassifier
)
from neuralk_foundry_ce.utils.data import make_deduplication
from neuralk_foundry_ce.datasets import get_data_config, LocalDataConfig
from dataclasses import dataclass


def copy_splits(source_root: Path, target_root: Path):
    """
    Copy all `splits.json` files from source_root to target_root,
    preserving the relative directory structure under source_root.

    Parameters
    ----------
    source_root : Path
        Source root directory (e.g., Path("cache_xgboost"))
    target_root : Path
        Destination root directory (e.g., Path("cache"))
    """
    pattern = str(source_root / "*/fold_*/1_stratified-shuffle-split/splits.json")
    for src_path in glob.glob(pattern):
        src = Path(src_path)
        rel_path = src.relative_to(source_root)
        dst = target_root / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


if __name__ == '__main__':
    from tabbench.workflow.use_cases import Classification

    features, target = make_classification(random_state=42)
    df = pd.DataFrame(features)
    df['target'] = target
    df.to_parquet('./my_dataset.parquet')

    @dataclass
    class DataConfig(LocalDataConfig):
        name: str='fake_classification'
        task: str = "classification"
        target: str = 'target'
        file_path: str = "./my_dataset.parquet"

    dataset = 'fake_classification'

    # Check that the dataset is well imported
    get_data_config('fake_classification').name
    
    # Name, class, categorical_encoding, numerical_encoding
    models = [
        ('xgboost', XGBoostClassifier, 'integer', 'none'),
        ('catboost', CatBoostClassifier, 'integer', 'none'),
        ('lightgbm', LightGBMClassifier, 'integer', 'none'),
        ('mlp', MLPClassifier, 'integer', 'standard'),
        ('tabicl', TabICLClassifier, 'none', 'none'),
        ('tabpfn', TabPFNClassifier, 'none', 'none'),
    ]
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    reference_cache = script_dir / 'cache'

    for model_name, model_class, categorical_encoding, numerical_encoding in models:
        print('Model:', model_name)

        model_cache = script_dir / f'cache_{dataset}'
        if not model_cache.exists() and reference_cache.exists():
            # Copy the cache reference for the new model
            shutil.copytree(reference_cache, model_cache)

        print(f'Dataset {dataset}')

        try:
            for fold_index in range(5):
                fold_cache = model_cache / dataset / f'fold_{fold_index}'
                workflow = Classification(dataset, cache_dir=fold_cache)
                workflow.set_parameter('categorical_encoding', categorical_encoding)
                workflow.set_parameter('numerical_encoding', numerical_encoding)
                workflow.set_classifier(model_class())
                data, metrics = workflow.run(fold_index=fold_index)
                print(metrics[model_class.name]['test_roc_auc'])
                with open(fold_cache / 'results.json', 'w') as f:
                    json.dump(make_json_serializable(metrics), f)
        except Exception as e:
            print('FAILED')
            print(f'Error is: {e}')
            continue

        if not reference_cache.exists():
            # We use the generated splits to create a reference
            copy_splits(model_cache, reference_cache)

        print('SUCCESS')
