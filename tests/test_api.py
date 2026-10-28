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


class _PassingReport:
    def has_fails(self) -> bool:
        return False


class _FailingReport:
    def has_fails(self) -> bool:
        return True


def _traits_dataset() -> xr.Dataset:
    ds = xr.Dataset()
    ds.attrs["mlwp_time_trait"] = "forecast"
    ds.attrs["mlwp_space_trait"] = "grid"
    ds.attrs["mlwp_uncertainty_trait"] = "deterministic"
    return ds


def test_callable_loader_is_used_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A callable loader receives the path and kwargs and its dataset is returned."""
    monkeypatch.setattr(
        "mlwp_data_loaders.api.validate_dataset",
        lambda *args, **kwargs: _PassingReport(),
    )
    calls: list[tuple] = []

    def loader(path, **kwargs):
        calls.append((path, kwargs))
        return _traits_dataset()

    ds = load_and_validate_dataset(
        "dummy.zarr", loader=loader, storage_options={"anon": True}
    )

    assert isinstance(ds, xr.Dataset)
    assert calls == [("dummy.zarr", {"storage_options": {"anon": True}})]


def test_callable_loader_returns_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """return_validation_report works with a callable loader."""
    monkeypatch.setattr(
        "mlwp_data_loaders.api.validate_dataset",
        lambda *args, **kwargs: _FailingReport(),
    )

    ds, report = load_and_validate_dataset(
        "dummy.zarr",
        loader=lambda path, **kwargs: _traits_dataset(),
        return_validation_report=True,
    )  # type: ignore

    assert isinstance(ds, xr.Dataset)
    assert report.has_fails()


def test_callable_loader_validation_failure_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing validation raises ValueError for callable loaders too."""
    monkeypatch.setattr(
        "mlwp_data_loaders.api.validate_dataset",
        lambda *args, **kwargs: _FailingReport(),
    )

    with pytest.raises(ValueError, match="Dataset validation failed"):
        load_and_validate_dataset(
            "dummy.zarr", loader=lambda path, **kwargs: _traits_dataset()
        )


def test_loader_of_wrong_type_raises_typeerror() -> None:
    """A loader that is neither a string nor a callable is rejected."""
    with pytest.raises(TypeError, match="loader must be a string or a callable"):
        load_and_validate_dataset("dummy.zarr", loader=123)  # type: ignore[arg-type]
