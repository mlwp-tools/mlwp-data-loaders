"""Tests for the mlwp-data-loaders Python API."""

from __future__ import annotations

import pytest
import xarray as xr

import mlwp_data_loaders.mxalign_api as mxalign_api
from mlwp_data_loaders.api import load_dataset
from mlwp_data_loaders.core import get_dataset_traits_from_loader


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


def test_get_dataset_traits_from_loader_raises_missing_load_dataset(tmp_path) -> None:
    """Loader modules must define a 'load_dataset' function."""
    loader_file = tmp_path / "loader_missing.py"
    loader_file.write_text("TIME_PROFILE = 'forecast'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must define a 'load_dataset' function"):
        get_dataset_traits_from_loader(str(loader_file))


def test_get_dataset_traits_from_loader_finds_constants(tmp_path) -> None:
    """Loader modules can define trait constants which are correctly captured."""
    loader_file = tmp_path / "loader_valid.py"
    loader_file.write_text(
        "def load_dataset(path, **kwargs): return None\n"
        "TIME_PROFILE = 'forecast'\n"
        "SPACE_PROFILE = 'grid'\n",
        encoding="utf-8",
    )
    traits = get_dataset_traits_from_loader(str(loader_file))
    assert "load_dataset" in traits
    assert traits["time_profile"] == "forecast"
    assert traits["space_profile"] == "grid"
    assert "uncertainty_profile" not in traits


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


def test_validate_dataset_with_mxalign_returns_fail_report_for_invalid_dims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mxalign validation failures are converted into report entries."""

    def mock_load_symbols():
        def mock_validate(ds, props):
            raise ValueError("Mock mxalign validation failed")

        return {
            "Properties": lambda time, space, uncertainty: "props",
            "Space": lambda x: x,
            "Time": lambda x: x,
            "Uncertainty": lambda x: x,
            "validate_dataset": mock_validate,
        }

    monkeypatch.setattr(
        "mlwp_data_loaders.mxalign_api._load_mxalign_validation_symbols",
        mock_load_symbols,
    )

    report = mxalign_api.validate_dataset_with_mxalign(
        _forecast_grid_ds(),
        time="observation",
        space="point",
        uncertainty="deterministic",
    )

    assert report.has_fails()
    assert len(report.results) == 1
    assert report.results[0].section == "MXAlign Properties"
