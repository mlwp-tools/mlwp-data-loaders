"""Helpers for validating datasets with local mxalign property checks."""

from __future__ import annotations

from functools import lru_cache

import xarray as xr
from mlwp_data_specs.specs.reporting import ValidationReport


@lru_cache(maxsize=1)
def _load_mxalign_validation_symbols():
    """Load mxalign property validation modules from the installed package."""
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
    """Validate a dataset with mxalign property checks when selectors are known."""
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
