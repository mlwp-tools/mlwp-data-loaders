"""Integration tests for the built-in ``anemoi-datasets`` loader."""

from __future__ import annotations

from mlwp_data_specs import validate_dataset

from mlwp_data_loaders.api import load_dataset
from mlwp_data_loaders.mxalign_api import validate_dataset_with_mxalign

# Use small CERRA sample dataset stored on EWC (European Weather Cloud)
# S3-compatible object store for testing.
DATASET_PATH = (
    "s3://mlwp-sample-datasets/anemoi-datasets/"
    "cerra-rr-an-oper-0001-mars-5p5km-2017-2017-6h-v3-testing.zarr/"
)
ENDPOINT_URL = "https://object-store.os-api.cci2.ecmwf.int"
LOADER = "mlwp_data_loaders.loaders.anemoi.anemoi_datasets"


def test_load_dataset_opens_anemoi_store_from_ewc() -> None:
    """The anemoi-datasets loader can open and validate the sample Zarr store."""
    storage_options: dict[str, object] = {
        "endpoint_url": ENDPOINT_URL,
        "anon": True,
    }

    ds = load_dataset(
        DATASET_PATH,
        loader=LOADER,
        storage_options=storage_options,
    )
    report = validate_dataset(
        ds,
        time="observation",
        space="point",
        uncertainty="deterministic",
    )
    report += validate_dataset_with_mxalign(
        ds,
        time="observation",
        space="point",
        uncertainty="deterministic",
    )

    if report.has_fails():
        report.console_print()
    assert not report.has_fails()
