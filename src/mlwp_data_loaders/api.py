"""Public Python API for loading and validating datasets."""

from __future__ import annotations

import inspect
from typing import Any

import xarray as xr
from mlwp_data_specs import validate_dataset

from .core import get_dataset_traits_from_loader


def load_dataset(
    dataset_path: str | list[str],
    *,
    loader: str,
    storage_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> xr.Dataset:
    """Load a dataset through a loader module and validate it.

    Parameters
    ----------
    dataset_path : str | list[str]
        One path or a list of paths to source datasets.
    loader : str
        Loader module reference. A value ending in ``.py`` is treated as a file
        path. A value containing ``.`` is treated as a Python module path.
    storage_options : dict[str, Any] | None, optional
        Storage options forwarded to the loader's ``load_dataset`` function.
    **kwargs
        Additional keyword arguments forwarded to the loader's ``load_dataset``
        function if its signature accepts them.

    Returns
    -------
    xr.Dataset
        Loaded and validated dataset.
    """
    traits = get_dataset_traits_from_loader(loader)

    loader_func = traits["load_dataset"]
    sig = inspect.signature(loader_func)

    loader_kwargs: dict[str, Any] = {}

    # Check if the loader's load_dataset accepts **kwargs
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
    )

    if storage_options is not None:
        if accepts_kwargs or "storage_options" in sig.parameters:
            loader_kwargs["storage_options"] = storage_options

    for key, value in kwargs.items():
        if accepts_kwargs or key in sig.parameters:
            loader_kwargs[key] = value

    ds = loader_func(dataset_path, **loader_kwargs)

    if not isinstance(ds, xr.Dataset):
        ds = ds.to_dataset()

    validate_dataset(
        ds,
        time=traits.get("time_profile"),
        space=traits.get("space_profile"),
        uncertainty=traits.get("uncertainty_profile"),
    )

    return ds
