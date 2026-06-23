"""Integration tests for the IFS forecast loader."""

from __future__ import annotations

from pathlib import Path

import fsspec
import pytest
from botocore.exceptions import BotoCoreError
from mlwp_data_specs.api import (
    SPACE_TRAIT_ATTR,
    TIME_TRAIT_ATTR,
    UNCERTAINTY_TRAIT_ATTR,
)

from mlwp_data_loaders.api import load_and_validate_dataset
from mlwp_data_loaders.mxalign_api import validate_dataset_with_mxalign

pytest.importorskip("cfgrib")

LOADER = "mlwp_data_loaders.loaders.ifs.forecast"

ENDPOINT_URL = "https://object-store.os-api.cci2.ecmwf.int"
STORAGE_OPTS = {"endpoint_url": ENDPOINT_URL, "anon": True}

BUCKET = "mlwp-sample-datasets/ifs/2026-06-19"
DET_KEYS = [
    f"{BUCKET}/deterministic/ifs_det_fcst_20260101.grib",
    f"{BUCKET}/deterministic/ifs_det_fcst_20260102.grib",
]
ENS_KEYS = [
    f"{BUCKET}/ensemble/ifs_ens_fcst_20260101.grib",
    f"{BUCKET}/ensemble/ifs_ens_fcst_20260102.grib",
]


@pytest.fixture
def download_grib_files(tmp_path):
    """Return a helper that downloads S3 GRIB keys to a temp dir.

    cfgrib reads GRIB via eccodes, which requires a local file path and cannot
    stream from object storage, so the sample files are fetched locally first.
    """

    fs = fsspec.filesystem("s3", **STORAGE_OPTS)

    def _download(keys: list[str]) -> list[str]:
        local_paths = []
        for key in keys:
            dest = tmp_path / Path(key).name
            try:
                fs.get(key, str(dest))
            except (OSError, BotoCoreError) as exc:
                pytest.skip(f"Could not reach object storage ({ENDPOINT_URL}): {exc}")
            local_paths.append(str(dest))
        return local_paths

    return _download


@pytest.mark.network
@pytest.mark.parametrize(
    "keys, expected_uncertainty",
    [
        pytest.param(DET_KEYS, "deterministic", id="deterministic"),
        pytest.param(ENS_KEYS, "ensemble", id="ensemble"),
    ],
)
def test_load_dataset_opens_ifs_grib(
    download_grib_files, keys, expected_uncertainty
) -> None:
    """The IFS forecast loader can open and validate GRIB files."""

    paths = download_grib_files(keys)

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
