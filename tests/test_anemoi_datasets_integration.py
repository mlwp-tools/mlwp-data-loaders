"""Integration tests for the built-in ``anemoi-datasets`` loader."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from mlwp_data_loaders.api import load_dataset

# Use small CERRA sample dataset stored on EWC (European Weather Cloud)
# S3-compatible object store for testing.
DATASET_PATH = (
    "s3://mlwp-sample-datasets/anemoi-datasets/"
    "cerra-rr-an-oper-0001-mars-5p5km-2017-2017-6h-v3-testing.zarr/"
)
ENDPOINT_URL = "https://object-store.os-api.cci2.ecmwf.int"
LOADER = "mlwp_data_loaders.loaders.anemoi.anemoi_datasets"


def test_load_dataset_opens_anemoi_store_from_ewc() -> None:
    """The anemoi-datasets loader can open the sample Zarr store from S3."""
    storage_options: dict[str, object] = {
        "endpoint_url": ENDPOINT_URL,
        "anon": True,
    }

    try:
        ds = load_dataset(
            DATASET_PATH,
            loader=LOADER,
            storage_options=storage_options,
        )
    except Exception as exc:  # pragma: no cover - environment-dependent fallback
        reason = f"{exc.__class__.__module__}.{exc.__class__.__name__}: {exc}"
        known_access_failures = (
            "AccessDenied",
            "CERTIFICATE_VERIFY_FAILED",
            "Cannot connect to host",
            "SSL validation failed",
            "Temporary failure in name resolution",
            "Unable to locate credentials",
        )
        if exc.__class__.__name__ in {
            "ClientConnectorCertificateError",
            "NoCredentialsError",
            "PermissionError",
        } or any(token in str(exc) for token in known_access_failures):
            pytest.skip(f"S3 integration environment unavailable: {reason}")
        raise

    assert isinstance(ds, xr.Dataset)
    assert "valid_time" in ds.dims
    assert "grid_index" in ds.dims
    assert ds.sizes["valid_time"] > 0
    assert ds.sizes["grid_index"] > 0
    assert "ensemble" not in ds.dims
    assert "variable" not in ds.dims
    assert "time" not in ds.dims
    assert {"latitude", "longitude", "valid_time"} <= set(ds.coords)
    assert np.issubdtype(ds.coords["valid_time"].dtype, np.datetime64)
    assert ds.coords["latitude"].dtype == np.float32
    assert ds.coords["longitude"].dtype == np.float32
    assert "data" not in ds.data_vars
    assert len(ds.data_vars) > 0
