"""Tests for the mlwp-data-loaders Python API."""

from __future__ import annotations

import pytest
import xarray as xr

from mlwp_data_loaders.api import load_dataset
from mlwp_data_loaders.core import import_loader_hooks


def _forecast_grid_ds() -> xr.Dataset:
    """Create a forecast + grid dataset for loader tests."""
    ds = xr.Dataset(
        coords={
            "reference_time": ("reference_time", [0]),
            "lead_time": ("lead_time", [1]),
            "longitude": ("longitude", [10.0, 11.0]),
            "latitude": ("latitude", [60.0, 61.0]),
        }
    )
    ds.coords["reference_time"].attrs["standard_name"] = "forecast_reference_time"
    ds.coords["lead_time"].attrs.update(
        {"standard_name": "forecast_period", "units": "hours"}
    )
    ds.coords["longitude"].attrs.update(
        {"standard_name": "longitude", "units": "degrees_east"}
    )
    ds.coords["latitude"].attrs.update(
        {"standard_name": "latitude", "units": "degrees_north"}
    )
    return ds


def test_import_loader_hooks_defaults_from_module(tmp_path) -> None:
    """Loader modules may omit hooks."""
    loader_file = tmp_path / "loader_defaults.py"
    loader_file.write_text("", encoding="utf-8")
    assert import_loader_hooks(str(loader_file)) == {}


def test_load_dataset_uses_loader_hooks(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Datasets are opened through the configured loader module."""
    loader_file = tmp_path / "loader_hooks.py"
    loader_file.write_text(
        "\n".join(
            [
                "CONCAT_DIM = 'sample'",
                "def preprocess(ds):",
                "    return ds.rename({'reference_time': 'sample'})",
                "def postprocess(ds):",
                "    return ds.assign_coords(source=('sample', ['loader']))",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(xr, "open_dataset", lambda *args, **kwargs: _forecast_grid_ds())

    ds = load_dataset("dummy.nc", loader=str(loader_file))
    assert "sample" in ds.dims
    assert ds.coords["source"].item() == "loader"


def test_load_dataset_rejects_incompatible_loader_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trait selections incompatible with the loader are rejected."""
    monkeypatch.setattr(xr, "open_dataset", lambda *args, **kwargs: _forecast_grid_ds())

    with pytest.raises(ValueError, match="Loader does not support time='observation'"):
        load_dataset(
            ["a.nc", "b.nc"],
            loader="mlwp_data_loaders.loaders.anemoi.anemoi_inference",
            time="observation",
        )


def test_load_dataset_requires_concat_dim_for_multiple_inputs(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multiple paths require an explicit concat dimension."""
    loader_file = tmp_path / "loader_without_concat.py"
    loader_file.write_text("", encoding="utf-8")
    monkeypatch.setattr(xr, "open_dataset", lambda *args, **kwargs: _forecast_grid_ds())

    with pytest.raises(ValueError, match="Loader must define 'concat_dim'"):
        load_dataset(["a.nc", "b.nc"], loader=str(loader_file), time="forecast")
