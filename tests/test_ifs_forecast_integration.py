"""Integration tests for the IFS forecast loader."""

from __future__ import annotations

import pooch
import pytest
from mlwp_data_specs.api import (
    SPACE_TRAIT_ATTR,
    TIME_TRAIT_ATTR,
    UNCERTAINTY_TRAIT_ATTR,
)

from mlwp_data_loaders.api import load_and_validate_dataset
from mlwp_data_loaders.mxalign_api import validate_dataset_with_mxalign

pytest.importorskip("cfgrib")

LOADER = "mlwp_data_loaders.loaders.ifs.forecast"

BASE_URL = (
    "https://object-store.os-api.cci2.ecmwf.int/mlwp-sample-datasets/ifs/2026-06-19"
)

DET_FILES = {
    "deterministic/ifs_det_fcst_20260101.grib": "md5:6c65edd213a23dc423dcee7c88b84183",
    "deterministic/ifs_det_fcst_20260102.grib": "md5:f4cfe072eb37482b606cb16684844bca",
}
ENS_FILES = {
    "ensemble/ifs_ens_fcst_20260101.grib": "md5:392ae6742ad66c7a8b1c960c1131e1bf",
    "ensemble/ifs_ens_fcst_20260102.grib": "md5:8b46b1e4c8a0d86ecb79723e38a570f3",
}


@pytest.fixture
def download_grib_files():
    """Return a helper that downloads and caches the sample GRIB files.

    cfgrib reads GRIB via eccodes, which requires a local file path and cannot
    stream from object storage, so the sample files are fetched (and cached)
    locally first via pooch.
    """

    def _download(files: dict[str, str]) -> list[str]:
        local_paths = []
        for name, known_hash in files.items():
            try:
                path = pooch.retrieve(
                    url=f"{BASE_URL}/{name}",
                    known_hash=known_hash,
                )
            except OSError as exc:
                pytest.skip(f"Could not download sample GRIB file ({name}): {exc}")
            local_paths.append(path)
        return local_paths

    return _download


@pytest.mark.network
@pytest.mark.parametrize(
    "files, expected_uncertainty",
    [
        pytest.param(DET_FILES, "deterministic", id="deterministic"),
        pytest.param(ENS_FILES, "ensemble", id="ensemble"),
    ],
)
def test_load_dataset_opens_ifs_grib(
    download_grib_files, files, expected_uncertainty
) -> None:
    """The IFS forecast loader can open and validate GRIB files."""

    paths = download_grib_files(files)

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
