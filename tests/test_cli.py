"""Tests for the mlwp-data-loaders CLI."""

from __future__ import annotations

import xarray as xr
from pytest import MonkeyPatch

from mlwp_data_loaders import cli


def _forecast_grid_ds() -> xr.Dataset:
    """Create a minimal forecast + grid dataset for CLI tests."""
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


def test_cli_accepts_multiple_dataset_paths(monkeypatch: MonkeyPatch) -> None:
    """CLI passes multiple dataset paths through to the load/validate API."""
    observed: dict[str, object] = {}

    def _load_dataset(dataset_path, **kwargs):
        observed["dataset_path"] = dataset_path
        return _forecast_grid_ds()

    def _get_dataset_traits_from_loader(loader):
        return {
            "time_profile": "forecast",
            "space_profile": "grid",
            "uncertainty_profile": "deterministic",
        }

    class _Report:
        def __init__(self):
            self.fails = False

        def console_print(self):
            return None

        def has_fails(self):
            return self.fails

        def __iadd__(self, other):
            return self

    def _validate_dataset(ds, **kwargs):
        return _Report()

    monkeypatch.setattr(cli, "load_dataset", _load_dataset)
    monkeypatch.setattr(
        cli, "get_dataset_traits_from_loader", _get_dataset_traits_from_loader
    )
    monkeypatch.setattr(cli, "validate_dataset", _validate_dataset)
    monkeypatch.setattr(
        cli, "validate_dataset_with_mxalign", lambda *args, **kwargs: _Report()
    )

    code = cli.main(
        [
            "a.nc",
            "b.nc",
            "--loader",
            "mlwp_data_loaders.loaders.anemoi.anemoi_inference",
        ]
    )

    assert code == 0
    assert observed["dataset_path"] == ["a.nc", "b.nc"]
