"""Tests for the built-in ``anemoi-inference`` loader on small local files."""

from __future__ import annotations

import h5netcdf
import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loguru import logger

from mlwp_data_loaders.api import load_and_validate_dataset
from mlwp_data_loaders.loaders.anemoi.anemoi_inference import load_dataset

LOADER = "mlwp_data_loaders.loaders.anemoi.anemoi_inference"
REFERENCE_TIMES = pd.date_range("2020-02-01", periods=4, freq="12h")
N_STEPS = 5
N_CELLS = 3


def _write_forecast(
    path, reference_time, n_steps=N_STEPS, variables=("2t", "msl")
) -> str:
    """Write a file laid out like anemoi-inference NetCDF output.

    ``time`` is unlimited and ``lsm``, a constant field, holds only the first
    step, as anemoi-inference writes it.
    """
    rng = np.random.default_rng(reference_time.value)
    fill = np.float32(np.nan)
    with h5netcdf.File(path, "w") as f:
        f.dimensions = {"time": None, "values": N_CELLS}
        f.resize_dimension("time", n_steps)
        time = f.create_variable(
            "time", ("time",), "i4", data=np.arange(n_steps) * 6 * 3600
        )
        time.attrs.update(
            {"units": f"seconds since {reference_time}", "calendar": "gregorian"}
        )
        for name, values in [
            ("latitude", [50.0, 51.0, 52.0]),
            ("longitude", [3, 4, 5]),
        ]:
            coord = f.create_variable(name, ("values",), "f4", data=values)
            coord.attrs["long_name"] = name
        for name in [*variables, "lsm"]:
            var = f.create_variable(
                name,
                ("time", "values"),
                "f4",
                data=rng.random((n_steps, N_CELLS), dtype=np.float32),
                fillvalue=fill,
            )
            var.attrs["missing_value"] = fill
    with h5py.File(path, "a") as f:
        f["lsm"].resize((1, N_CELLS))
    return str(path)


@pytest.fixture
def paths(tmp_path) -> list[str]:
    return [_write_forecast(tmp_path / f"{t:%Y%m%d%H}.nc", t) for t in REFERENCE_TIMES]


@pytest.fixture
def warnings():
    messages = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(sink)


def test_load_dataset_matches_open_dataset(paths) -> None:
    """Each file is one reference time with the values xarray reads from it."""
    ds = load_dataset(paths)

    assert list(ds.data_vars) == ["2t", "msl", "lsm"]
    assert ds["2t"].dims == ("reference_time", "lead_time", "grid_index")
    assert ds["2t"].chunks == ((1,) * len(paths), (N_STEPS,), (N_CELLS,))
    np.testing.assert_array_equal(ds["reference_time"].values, REFERENCE_TIMES.values)
    np.testing.assert_array_equal(
        ds["lead_time"].values, pd.timedelta_range("0h", periods=N_STEPS, freq="6h")
    )
    for i, path in enumerate(paths):
        expected = xr.open_dataset(path, engine="h5netcdf")
        for name in ds.data_vars:
            xr.testing.assert_identical(
                ds[name]
                .isel(reference_time=i)
                .drop_vars(["reference_time", "lead_time", "latitude", "longitude"]),
                expected[name]
                .drop_vars("time")
                .rename(time="lead_time", values="grid_index"),
            )
    # The constant field is padded with its fill value after the first step
    assert np.isnan(ds["lsm"].values[:, 1:]).all()
    assert ds["latitude"].attrs == {
        "long_name": "latitude",
        "standard_name": "latitude",
        "units": "degrees_north",
    }


def test_load_dataset_selects_variables(paths) -> None:
    """A single name or a list selects those variables, unknown names raise."""
    ds = load_dataset(paths, variables="msl")
    assert list(ds.data_vars) == ["msl"]

    with pytest.raises(ValueError, match="No .* variables \\['tp'\\]"):
        load_dataset(paths, variables=["2t", "tp"])


def test_load_dataset_skips_files_that_differ(tmp_path, paths, warnings) -> None:
    """Files with other lead times or missing variables are skipped, with a warning.

    The lead times kept are the most common ones, even when the first file
    differs.
    """
    short = _write_forecast(tmp_path / "short.nc", REFERENCE_TIMES[0], n_steps=1)
    no_msl = _write_forecast(
        tmp_path / "no_msl.nc", REFERENCE_TIMES[0], variables=("2t",)
    )

    ds = load_dataset([short, *paths, no_msl])

    np.testing.assert_array_equal(ds["reference_time"].values, REFERENCE_TIMES.values)
    assert ds.sizes["lead_time"] == N_STEPS
    assert len(warnings) == 2
    assert short in warnings[0] and "lead times" in warnings[0]
    assert no_msl in warnings[1] and "['msl']" in warnings[1]


def test_load_dataset_graph_does_not_grow_with_files(paths) -> None:
    """One Blockwise layer per variable, whatever the number of files."""
    layers = [len(load_dataset(p).__dask_graph__().layers) for p in [paths[:1], paths]]
    assert layers == [4, 4]

    graph = load_dataset(paths).__dask_graph__()
    # Only the layer holding the file paths is materialized, one task per file
    assert [
        len(layer) for layer in graph.layers.values() if layer.is_materialized()
    ] == [len(paths)]


def test_load_and_validate_dataset_passes_validation(paths) -> None:
    """The loaded dataset passes the mlwp-data-specs validation."""
    _, report = load_and_validate_dataset(
        paths, loader=LOADER, return_validation_report=True
    )
    if report.has_fails():
        report.console_print()
    assert not report.has_fails()
