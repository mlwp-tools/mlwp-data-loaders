"""Tests for the mlwp-data-loaders Python API."""

from __future__ import annotations

import pytest
import xarray as xr

from mlwp_data_loaders.api import load_and_validate_dataset
from mlwp_data_loaders.core import get_loader_func


def test_get_loader_func_raises_missing_load_dataset(tmp_path) -> None:
    """Loader modules must define a 'load_dataset' function."""
    loader_file = tmp_path / "loader_missing.py"
    loader_file.write_text("TIME_PROFILE = 'forecast'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must define a 'load_dataset' function"):
        get_loader_func(str(loader_file))


def test_load_dataset_rejects_unsupported_kwargs(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check that api.load_dataset rejects kwargs not supported by the loader."""
    loader_file = tmp_path / "loader_strict.py"
    loader_file.write_text(
        "def load_dataset(path, chunks=None):\n"
        "    from xarray import Dataset\n"
        "    ds = Dataset()\n"
        "    ds.attrs['chunks'] = chunks\n"
        "    ds.attrs['mlwp_time_trait'] = 'forecast'\n"
        "    ds.attrs['mlwp_space_trait'] = 'grid'\n"
        "    ds.attrs['mlwp_uncertainty_trait'] = 'deterministic'\n"
        "    return ds\n",
        encoding="utf-8",
    )

    class MockReport:
        def has_fails(self):
            return False

    # Mock validate_dataset to bypass validation on an empty dataset
    monkeypatch.setattr(
        "mlwp_data_loaders.api.validate_dataset", lambda *args, **kwargs: MockReport()
    )

    with pytest.raises(TypeError, match="unexpected keyword argument 'engine'"):
        load_and_validate_dataset(
            "dummy.nc",
            loader=str(loader_file),
            chunks="auto",
            engine="h5netcdf",  # Should raise TypeError
        )


def test_load_dataset_returns_report(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that load_and_validate_dataset returns report when requested."""
    loader_file = tmp_path / "loader_traits.py"
    loader_file.write_text(
        "def load_dataset(path, **kwargs):\n"
        "    from xarray import Dataset\n"
        "    ds = Dataset()\n"
        "    ds.attrs['mlwp_time_trait'] = 'forecast'\n"
        "    ds.attrs['mlwp_space_trait'] = 'grid'\n"
        "    ds.attrs['mlwp_uncertainty_trait'] = 'deterministic'\n"
        "    return ds\n",
        encoding="utf-8",
    )

    class MockReport:
        def has_fails(self):
            return False

    monkeypatch.setattr(
        "mlwp_data_loaders.api.validate_dataset", lambda *args, **kwargs: MockReport()
    )

    res = load_and_validate_dataset(
        "dummy.nc",
        loader=str(loader_file),
        return_validation_report=True,
    )
    assert isinstance(res, tuple)

    ds, report = res  # type: ignore
    assert isinstance(ds, xr.Dataset)
    assert not report.has_fails()
    assert ds.attrs.get("mlwp_time_trait") == "forecast"
