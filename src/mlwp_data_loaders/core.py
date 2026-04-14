"""Core loader import and dataset opening helpers."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any, Callable

_TRAIT_ATTR_FORMAT = "mlwp_{}_trait"

TIME_TRAIT_ATTR = _TRAIT_ATTR_FORMAT.format("time")
SPACE_TRAIT_ATTR = _TRAIT_ATTR_FORMAT.format("space")
UNCERTAINTY_TRAIT_ATTR = _TRAIT_ATTR_FORMAT.format("uncertainty")


def get_loader_func(loader: str) -> Callable[..., Any]:
    """Get the load_dataset function from a loader module.

    Parameters
    ----------
    loader : str
        Loader reference. A value ending in ``.py`` is treated as a file path.
        A value containing ``.`` is treated as a Python module path.

    Returns
    -------
    Callable
        The load_dataset function.

    Raises
    ------
    ValueError
        If the loader reference cannot be resolved or if the loader module
        does not define a 'load_dataset' function.
    """
    if loader.endswith(".py"):
        path = Path(loader)
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not import loader module from file: {loader}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    elif "." in loader:
        module = importlib.import_module(loader)
    else:
        raise ValueError(
            "Loader must be a Python file path ending in .py or a Python module path"
        )

    if not hasattr(module, "load_dataset"):
        raise ValueError(
            f"Loader module {loader!r} must define a 'load_dataset' function."
        )
    return module.load_dataset
