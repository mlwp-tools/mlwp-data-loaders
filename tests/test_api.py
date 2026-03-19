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


def test_import_loader_hooks_raises_missing_load_dataset(tmp_path) -> None:
    """Loader modules must define a 'load_dataset' function."""
    loader_file = tmp_path / "loader_missing.py"
    loader_file.write_text("TIME_PROFILE = 'forecast'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must define a 'load_dataset' function"):
        import_loader_hooks(str(loader_file))


def test_import_loader_hooks_finds_constants(tmp_path) -> None:
    """Loader modules can define trait constants which are correctly captured."""
    loader_file = tmp_path / "loader_valid.py"
    loader_file.write_text(
        "def load_dataset(path, **kwargs): return None\n"
        "TIME_PROFILE = 'forecast'\n"
        "SPACE_PROFILE = 'grid'\n",
        encoding="utf-8",
    )
    hooks = import_loader_hooks(str(loader_file))
    assert "load_dataset" in hooks
    assert hooks["time_profile"] == "forecast"
    assert hooks["space_profile"] == "grid"
    assert "uncertainty_profile" not in hooks


def test_load_dataset_filters_kwargs(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that api.load_dataset filters kwargs based on loader signature."""
    loader_file = tmp_path / "loader_strict.py"
    loader_file.write_text(
        "def load_dataset(path, chunks=None):\n"
        "    from xarray import Dataset\n"
        "    ds = Dataset()\n"
        "    ds.attrs['chunks'] = chunks\n"
        "    return ds\n"
        "TIME_PROFILE = 'forecast'\n",
        encoding="utf-8",
    )

    # Mock validate_dataset to bypass validation on an empty dataset
    monkeypatch.setattr(
        "mlwp_data_loaders.api.validate_dataset", lambda *args, **kwargs: None
    )

    ds = load_dataset(
        "dummy.nc",
        loader=str(loader_file),
        chunks="auto",
        engine="h5netcdf",  # Should be ignored because strict load_dataset doesn't take 'engine'
        storage_options={"anon": True},  # Should be ignored
    )

    assert ds.attrs["chunks"] == "auto"
    assert "engine" not in ds.attrs
