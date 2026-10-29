"""Tests for the built-in ``anemoi-datasets`` loader on a small local store."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mlwp_data_loaders.api import load_and_validate_dataset
from mlwp_data_loaders.loaders.anemoi.anemoi_datasets import load_dataset

LOADER = "mlwp_data_loaders.loaders.anemoi.anemoi_datasets"
VARIABLES = ["10u", "2t", "cos_latitude", "msl"]
DATES = pd.date_range("2020-01-01", periods=6, freq="3h")
N_CELLS = 5


def _write_store(path, time_chunk: int = 1) -> np.ndarray:
    """Write an anemoi-datasets-like Zarr store and return its ``data`` values."""
    data = np.arange(len(DATES) * len(VARIABLES) * N_CELLS, dtype=np.float32).reshape(
        len(DATES), len(VARIABLES), 1, N_CELLS
    )
    ds = xr.Dataset(
        {
            "data": (("time", "variable", "ensemble", "cell"), data),
            "dates": ("time", DATES.values),
            "latitudes": ("cell", np.linspace(40.0, 60.0, N_CELLS)),
            "longitudes": ("cell", np.linspace(-10.0, 30.0, N_CELLS)),
        },
        attrs={"variables": VARIABLES},
    )
    ds.to_zarr(
        path,
        consolidated=False,
        encoding={"data": {"chunks": (time_chunk, len(VARIABLES), 1, N_CELLS)}},
    )
    return data


@pytest.fixture
def store(tmp_path) -> tuple[str, np.ndarray]:
    """Write a small anemoi-datasets store and return its path and data."""
    path = str(tmp_path / "anemoi.zarr")
    return path, _write_store(path)


def test_load_dataset_drops_forcings_and_matches_store(store) -> None:
    """By default all non-forcing variables are loaded with the store's values."""
    path, data = store
    ds = load_dataset(path)

    assert list(ds.data_vars) == ["10u", "2t", "msl"]
    assert ds["2t"].dims == ("valid_time", "grid_index")
    np.testing.assert_array_equal(ds["2t"].values, data[:, 1, 0, :])
    np.testing.assert_array_equal(ds["msl"].values, data[:, 3, 0, :])
    np.testing.assert_array_equal(ds["valid_time"].values, DATES.values)
    assert ds["latitude"].dtype == np.float32


def test_load_dataset_selects_variables(store) -> None:
    """A single name or a list selects those variables, unknown names raise."""
    path, data = store

    ds = load_dataset(path, variables="msl")
    assert list(ds.data_vars) == ["msl"]
    np.testing.assert_array_equal(ds["msl"].values, data[:, 3, 0, :])

    with pytest.raises(ValueError, match="not found"):
        load_dataset(path, variables=["2t", "tp"])


def test_load_dataset_selects_valid_times(store) -> None:
    """``valid_times`` is an inclusive label slice; an empty one raises."""
    path, data = store

    ds = load_dataset(path, valid_times=slice("2020-01-01T03", "2020-01-01T09"))
    np.testing.assert_array_equal(ds["valid_time"].values, DATES[1:4].values)
    np.testing.assert_array_equal(ds["2t"].values, data[1:4, 1, 0, :])

    with pytest.raises(ValueError, match="No time steps"):
        load_dataset(path, valid_times=slice("2021-01-01", "2021-12-31"))


def test_load_dataset_variables_share_one_lazy_read(store) -> None:
    """All variables are split from one shared read, without materialized layers."""
    path, _ = store
    ds = load_dataset(path)
    graph = ds.__dask_graph__()

    # Only the single task holding the backend array is materialized
    assert all(
        len(layer) <= 1 for layer in graph.layers.values() if layer.is_materialized()
    )
    read_layers = {
        frozenset(graph.dependencies[ds[name].data.name]) for name in ds.data_vars
    }
    assert len(read_layers) == 1


def test_load_dataset_follows_store_time_chunks(tmp_path) -> None:
    """Dask chunks hold one on-disk time chunk and the full grid."""
    path = str(tmp_path / "anemoi.zarr")
    _write_store(path, time_chunk=2)

    ds = load_dataset(path)
    assert ds["2t"].chunks == ((2, 2, 2), (N_CELLS,))


def test_load_and_validate_dataset_passes_validation(store) -> None:
    """The loaded dataset passes the mlwp-data-specs validation."""
    path, _ = store
    _, report = load_and_validate_dataset(
        path, loader=LOADER, return_validation_report=True
    )
    if report.has_fails():
        report.console_print()
    assert not report.has_fails()
