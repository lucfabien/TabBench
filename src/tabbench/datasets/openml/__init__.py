import importlib
import pkgutil
from neuralk_foundry_ce.datasets.industrial import best_buy_simple_categ

__all__ = []

# Automatically import all submodules and add them to __all__
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__, prefix=__name__ + "."):
    short_name = module_name.split('.')[-1]
    if short_name.startswith("_"):
        continue
    imported_module = importlib.import_module(module_name)
    __all__.append(short_name)