"""Core loader import and dataset opening helpers."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

LoaderHooks = dict[str, Any]


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


def import_loader_hooks(loader: str) -> LoaderHooks:
    """Import hooks from a loader module.

    Parameters
    ----------
    loader : str
        Loader module reference.

    Returns
    -------
    LoaderHooks
        Mapping with hook names normalized to lowercase.

    Raises
    ------
    ValueError
        If the loader module does not define a 'load_dataset' function.
    """
    module = _load_module(loader)
    hooks: LoaderHooks = {}

    if not hasattr(module, "load_dataset"):
        raise ValueError(
            f"Loader module {loader!r} must define a 'load_dataset' function."
        )
    hooks["load_dataset"] = module.load_dataset

    supported_hooks = (
        "TIME_PROFILE",
        "SPACE_PROFILE",
        "UNCERTAINTY_PROFILE",
    )
    for name in supported_hooks:
        if hasattr(module, name):
            hooks[name.lower()] = getattr(module, name)

    return hooks
