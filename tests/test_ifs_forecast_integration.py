"""Integration tests for the IFS forecast loader."""

from __future__ import annotations

import pytest

pytest.importorskip("cfgrib")
from mlwp_data_specs.api import (
    SPACE_TRAIT_ATTR,
    TIME_TRAIT_ATTR,
    UNCERTAINTY_TRAIT_ATTR,
)

from mlwp_data_loaders.api import load_and_validate_dataset
from mlwp_data_loaders.mxalign_api import validate_dataset_with_mxalign

DET_FILES = [
    "/scratch/cu0k/ifs-example/ifs_det_fcst_20260101.grib",
    "/scratch/cu0k/ifs-example/ifs_det_fcst_20260102.grib",
]
ENS_FILES = [
    "/scratch/cu0k/ifs-example/ifs_ens_fcst_20260101.grib",
    "/scratch/cu0k/ifs-example/ifs_ens_fcst_20260102.grib",
]
LOADER = "mlwp_data_loaders.loaders.ifs.forecast"


@pytest.mark.parametrize(
    "paths, expected_uncertainty",
    [
        pytest.param(DET_FILES, "deterministic", id="deterministic"),
        pytest.param(ENS_FILES, "ensemble", id="ensemble"),
    ],
)
def test_load_dataset_opens_ifs_grib(paths, expected_uncertainty) -> None:
    """The IFS forecast loader can open and validate local GRIB files."""
    ds, report_specs = load_and_validate_dataset(  # type: ignore
        paths,
        loader=LOADER,
        chunks=None,
        return_validation_report=True,
    )

    assert ds.attrs[TIME_TRAIT_ATTR] == "forecast"
    assert ds.attrs[SPACE_TRAIT_ATTR] == "grid"
    assert ds.attrs[UNCERTAINTY_TRAIT_ATTR] == expected_uncertainty

    report_mxalign = validate_dataset_with_mxalign(
        ds,
        time=ds.attrs.get(TIME_TRAIT_ATTR),
        space=ds.attrs.get(SPACE_TRAIT_ATTR),
        uncertainty=ds.attrs.get(UNCERTAINTY_TRAIT_ATTR),
    )
    if report_mxalign.has_fails():
        report_mxalign.console_print()
    assert not report_mxalign.has_fails()

    if report_specs.has_fails():
        report_specs.console_print()
    assert not report_specs.has_fails()
