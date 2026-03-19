"""Helpers for validating datasets with local mxalign property checks."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import xarray as xr
from mlwp_data_specs.specs.reporting import ValidationReport


@lru_cache(maxsize=1)
def _load_mxalign_validation_symbols() -> dict[str, Any]:
    """Load mxalign property validation modules from the installed package.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the loaded mxalign classes and functions.

    Raises
    ------
    ImportError
        If the mxalign package is not installed or the required modules cannot be loaded.
    """
    try:
        from mxalign.properties.properties import Properties, Space, Time, Uncertainty
        from mxalign.properties.validation import validate_dataset
    except ImportError as e:
        raise ImportError(f"mxalign package is not installed: {e}")

    return {
        "Properties": Properties,
        "Space": Space,
        "Time": Time,
        "Uncertainty": Uncertainty,
        "validate_dataset": validate_dataset,
    }


def validate_dataset_with_mxalign(
    ds: xr.Dataset | xr.DataArray,
    *,
    time: str | None = None,
    space: str | None = None,
    uncertainty: str | None = None,
) -> ValidationReport:
    """Validate a dataset with mxalign property checks when selectors are known.

    Parameters
    ----------
    ds : xr.Dataset | xr.DataArray
        The xarray dataset or data array to validate.
    time : str | None, optional
        The time profile to validate against.
    space : str | None, optional
        The space profile to validate against.
    uncertainty : str | None, optional
        The uncertainty profile to validate against. Defaults to "deterministic".

    Returns
    -------
    ValidationReport
        A validation report object containing the results of the mxalign checks.
    """
    report = ValidationReport()
    if time is None or space is None:
        return report

    try:
        mxalign = _load_mxalign_validation_symbols()
    except ImportError as e:
        report.add(
            "MXAlign Properties",
            "mxalign.properties.validation",
            "WARNING",
            f"mxalign not available for validation: {e}",
        )
        return report

    properties = mxalign["Properties"](
        time=mxalign["Time"](time),
        space=mxalign["Space"](space),
        uncertainty=mxalign["Uncertainty"](uncertainty or "deterministic"),
    )

    try:
        mxalign["validate_dataset"](ds, properties)
    except ValueError as exc:
        report.add(
            "MXAlign Properties",
            "mxalign.properties.validation.validate_dataset",
            "FAIL",
            str(exc),
        )
    else:
        report.add(
            "MXAlign Properties",
            "mxalign.properties.validation.validate_dataset",
            "PASS",
        )
    return report
