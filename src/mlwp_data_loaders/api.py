"""Public Python API for loading and validating datasets."""

from __future__ import annotations

from typing import Any

import xarray as xr

from .core import import_loader_hooks, open_with_loader, validate_loader_profiles


def load_dataset(
    dataset_path: str | list[str],
    *,
    loader: str,
    time: str | None = None,
    space: str | None = None,
    uncertainty: str | None = None,
    storage_options: dict[str, Any] | None = None,
) -> xr.Dataset | xr.DataArray:
    """Load a dataset through a loader module.

    Parameters
    ----------
    dataset_path : str | list[str]
        One path or a list of paths to source datasets.
    loader : str
        Loader module reference. A value ending in ``.py`` is treated as a file
        path. A value containing ``.`` is treated as a Python module path.
    time : str | None, optional
        Optional time trait selector used to verify loader compatibility.
    space : str | None, optional
        Optional space trait selector used to verify loader compatibility.
    uncertainty : str | None, optional
        Optional uncertainty trait selector used to verify loader compatibility.
    storage_options : dict[str, Any] | None, optional
        Storage options forwarded to :func:`xarray.open_dataset`.

    Returns
    -------
    xr.Dataset | xr.DataArray
        Loaded dataset-like object.
    """
    hooks = import_loader_hooks(loader)
    validate_loader_profiles(
        hooks,
        time=time,
        space=space,
        uncertainty=uncertainty,
    )
    return open_with_loader(
        dataset_path,
        hooks=hooks,
        storage_options=storage_options,
    )
