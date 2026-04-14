"""Public Python API for loading and validating datasets."""

from __future__ import annotations

from typing import Any

import xarray as xr
from mlwp_data_specs import validate_dataset
from mlwp_data_specs.specs.reporting import ValidationReport

from .core import get_loader_func


def load_and_validate_dataset(
    dataset_path: str | list[str],
    *,
    loader: str,
    return_validation_report: bool = False,
    **kwargs: Any,
) -> xr.Dataset | tuple[xr.Dataset, ValidationReport]:
    """Load a dataset through a loader module and validate it.

    Parameters
    ----------
    dataset_path : str | list[str]
        One path or a list of paths to source datasets.
    loader : str
        Loader module reference. A value ending in ``.py`` is treated as a file
        path. A value containing ``.`` is treated as a Python module path.
    return_validation_report : bool, optional
        If True, return a tuple containing the dataset and the validation report.
        Defaults to False.
    **kwargs
        Additional keyword arguments forwarded to the loader's ``load_dataset``
        function (e.g., ``storage_options``).

    Returns
    -------
    xr.Dataset | tuple[xr.Dataset, ValidationReport]
        Loaded and validated dataset. If `return_validation_report` is True,
        returns a tuple of (dataset, validation_report).

    Raises
    ------
    ValueError
        If validation fails and `return_validation_report` is False.
    """
    loader_func = get_loader_func(loader)

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
