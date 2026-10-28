"""Integration tests for the built-in ``anemoi-inference`` loader."""

from __future__ import annotations

import numpy as np

from mlwp_data_loaders.api import load_and_validate_dataset

DATASET_PATH = [
    "s3://mlwp-sample-datasets/anemoi-inference/unknown-revision/"
    "anemoi-inference-lam_2020020100.nc",
    "s3://mlwp-sample-datasets/anemoi-inference/unknown-revision/"
    "anemoi-inference-lam_2020020200.nc",
]
ENDPOINT_URL = "https://object-store.os-api.cci2.ecmwf.int"
LOADER = "mlwp_data_loaders.loaders.anemoi.anemoi_inference"


def test_load_dataset_opens_anemoi_inference_from_ewc() -> None:
    """The anemoi-inference loader can open and validate the sample NetCDF files."""
    storage_options: dict[str, object] = {
        "endpoint_url": ENDPOINT_URL,
        "anon": True,
    }

    ds, report_specs = load_and_validate_dataset(  # type: ignore
        DATASET_PATH,
        loader=LOADER,
        storage_options=storage_options,
        return_validation_report=True,
    )

    if report_specs.has_fails():
        report_specs.console_print()
    assert not report_specs.has_fails()

    # Values are read lazily, one file per task, through fsspec
    assert np.isfinite(ds["2t"].isel(reference_time=-1, lead_time=0).values).any()
