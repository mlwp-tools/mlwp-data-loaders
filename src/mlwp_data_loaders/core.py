"""Core loader import and dataset opening helpers."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

DatasetTraits = dict[str, Any]


def _load_module(loader: str) -> ModuleType:
    """Import a loader module from a Python file or module path.

    Parameters
    ----------
    loader : str
        Loader reference. A value ending in ``.py`` is treated as a file path.
        A value containing ``.`` is treated as a Python module path.

    Returns
    -------
    ModuleType
        Imported module object.

    Raises
    ------
    ValueError
        If the loader reference cannot be resolved.
    """
    if loader.endswith(".py"):
        path = Path(loader)
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not import loader module from file: {loader}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    if "." in loader:
        return importlib.import_module(loader)
    raise ValueError(
        "Loader must be a Python file path ending in .py or a Python module path"
    )


def get_dataset_traits_from_loader(loader: str) -> DatasetTraits:
    """Import traits from a loader module.

    Parameters
    ----------
    loader : str
        Loader module reference.

    Returns
    -------
    DatasetTraits
        Mapping with trait names normalized to lowercase.

    Raises
    ------
    ValueError
        If the loader module does not define a 'load_dataset' function.
    """
    module = _load_module(loader)
    traits: DatasetTraits = {}

    if not hasattr(module, "load_dataset"):
        raise ValueError(
            f"Loader module {loader!r} must define a 'load_dataset' function."
        )
    traits["load_dataset"] = module.load_dataset

    supported_traits = (
        "TIME_PROFILE",
        "SPACE_PROFILE",
        "UNCERTAINTY_PROFILE",
    )
    for name in supported_traits:
        if hasattr(module, name):
            traits[name.lower()] = getattr(module, name)

    return traits
