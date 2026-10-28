"""Public Python API for loading and validating datasets."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import xarray as xr
from mlwp_data_specs import validate_dataset
from mlwp_data_specs.specs.reporting import ValidationReport

from .core import get_loader_func


def load_and_validate_dataset(
    dataset_path: str | list[str],
    *,
    loader: str | Callable[..., xr.Dataset],
    return_validation_report: bool = False,
    **kwargs: Any,
) -> xr.Dataset | tuple[xr.Dataset, ValidationReport]:
    """Load a dataset through a loader and validate it.

    Parameters
    ----------
    dataset_path : str | list[str]
        One path or a list of paths to source datasets.
    loader : str | Callable[..., xr.Dataset]
        Loader reference. A string ending in ``.py`` is treated as a file path.
        A string containing ``.`` is treated as a Python module path. In both
        cases the module's ``load_dataset`` function is used. A callable is used
        directly, as ``loader(dataset_path, **kwargs)``. It must return an
        ``xr.Dataset`` with the trait attributes set, since the traits are read
        from the attributes during validation.
    return_validation_report : bool, optional
        If True, return a tuple containing the dataset and the validation report.
        Defaults to False.
    **kwargs
        Additional keyword arguments forwarded to the loader function (e.g.,
        ``storage_options``).

    Returns
    -------
    xr.Dataset | tuple[xr.Dataset, ValidationReport]
        Loaded and validated dataset. If `return_validation_report` is True,
        returns a tuple of (dataset, validation_report).

    Raises
    ------
    TypeError
        If `loader` is neither a string nor a callable.
    ValueError
        If validation fails and `return_validation_report` is False.
    """
    if callable(loader):
        loader_func = loader
    elif isinstance(loader, str):
        loader_func = get_loader_func(loader)
    else:
        raise TypeError(
            f"loader must be a string or a callable, got {type(loader).__name__}"
        )

    ds = loader_func(dataset_path, **kwargs)

    if not isinstance(ds, xr.Dataset):
        ds = ds.to_dataset()

    report = validate_dataset(ds)

    if return_validation_report:
        return ds, report

    if report.has_fails():
        # Ideally, we should be able to format the report nicely
        raise ValueError(
            "Dataset validation failed. Run with return_validation_report=True for details."
        )

    return ds
