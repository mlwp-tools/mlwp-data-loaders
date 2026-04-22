"""Unit tests for the ``anemoi_inference_xy`` loader."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from mlwp_data_specs.api import (
    SPACE_TRAIT_ATTR,
    TIME_TRAIT_ATTR,
    UNCERTAINTY_TRAIT_ATTR,
)

from mlwp_data_loaders.loaders.anemoi import anemoi_inference_xy


def _make_raw_dataset(
    reference_time: str = "2022-01-01",
    time_steps: list[str] | None = None,
    ny: int = 3,
    nx: int = 4,
) -> xr.Dataset:
    """Create a synthetic raw dataset that matches the new anemoi-inference format."""
    if time_steps is None:
        time_steps = ["2022-01-01T06:00:00", "2022-01-01T12:00:00"]

    times = np.array(time_steps, dtype="datetime64[ns]")
    ref_time = np.datetime64(reference_time, "ns")

    y = np.linspace(-1e6, 1e6, ny, dtype=np.float32)
    x = np.linspace(-1e6, 1e6, nx, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    lat = (yy / 1e6 * 45.0).astype(np.float64)
    lon = (xx / 1e6 * 90.0).astype(np.float64)

    shape = (len(times), ny, nx)
    rng = np.random.default_rng(42)

    ds = xr.Dataset(
        {
            "forecast_reference_time": ref_time,
            "projection": np.int32(0),
            "2t": (["time", "y", "x"], rng.random(shape).astype(np.float32)),
            "tp": (["time", "y", "x"], rng.random(shape).astype(np.float32)),
        },
        coords={
            "time": times,
            "y": y,
            "x": x,
            "latitude": (["y", "x"], lat),
            "longitude": (["y", "x"], lon),
        },
    )
    return ds


class TestPreprocess:
    def test_reference_time_becomes_dimension(self) -> None:
        ds = _make_raw_dataset()
        ds_out = anemoi_inference_xy._preprocess(ds)
        assert "reference_time" in ds_out.dims

    def test_lead_time_becomes_dimension(self) -> None:
        ds = _make_raw_dataset()
        ds_out = anemoi_inference_xy._preprocess(ds)
        assert "lead_time" in ds_out.dims
        assert "time" not in ds_out.dims

    def test_lead_time_values_are_correct(self) -> None:
        ref = "2022-01-01"
        steps = ["2022-01-01T06:00:00", "2022-01-01T12:00:00"]
        ds = _make_raw_dataset(reference_time=ref, time_steps=steps)
        ds_out = anemoi_inference_xy._preprocess(ds)

        expected = np.array(steps, dtype="datetime64[ns]") - np.datetime64(ref, "ns")
        np.testing.assert_array_equal(ds_out["lead_time"].values, expected)

    def test_forecast_reference_time_var_is_dropped(self) -> None:
        ds = _make_raw_dataset()
        ds_out = anemoi_inference_xy._preprocess(ds)
        assert "forecast_reference_time" not in ds_out

    def test_projection_var_is_dropped(self) -> None:
        ds = _make_raw_dataset()
        ds_out = anemoi_inference_xy._preprocess(ds)
        assert "projection" not in ds_out

    def test_time_coord_is_dropped(self) -> None:
        ds = _make_raw_dataset()
        ds_out = anemoi_inference_xy._preprocess(ds)
        assert "time" not in ds_out.coords

    def test_latitude_longitude_remain_as_coordinates(self) -> None:
        ds = _make_raw_dataset()
        ds_out = anemoi_inference_xy._preprocess(ds)
        assert "latitude" in ds_out.coords
        assert "longitude" in ds_out.coords

    def test_latitude_longitude_promoted_when_data_vars(self) -> None:
        """Loader must promote lat/lon if they arrive as data variables."""
        ds = _make_raw_dataset()
        # Move lat/lon from coords to data vars
        lat_data = ds.coords["latitude"].values
        lon_data = ds.coords["longitude"].values
        ds = ds.drop_vars(["latitude", "longitude"])
        ds["latitude"] = (["y", "x"], lat_data)
        ds["longitude"] = (["y", "x"], lon_data)

        ds_out = anemoi_inference_xy._preprocess(ds)
        assert "latitude" in ds_out.coords
        assert "longitude" in ds_out.coords

    def test_y_x_dimensions_preserved(self) -> None:
        ds = _make_raw_dataset()
        ds_out = anemoi_inference_xy._preprocess(ds)
        assert "y" in ds_out.dims
        assert "x" in ds_out.dims

    def test_weather_variables_preserved(self) -> None:
        ds = _make_raw_dataset()
        ds_out = anemoi_inference_xy._preprocess(ds)
        assert "2t" in ds_out.data_vars
        assert "tp" in ds_out.data_vars


class TestLoadDataset:
    def test_traits_are_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        preprocessed = anemoi_inference_xy._preprocess(_make_raw_dataset())

        monkeypatch.setattr(
            anemoi_inference_xy.xr,
            "open_mfdataset",
            lambda *a, **kw: preprocessed,
        )

        result = anemoi_inference_xy.load_dataset("dummy.nc")
        assert result.attrs[TIME_TRAIT_ATTR] == "forecast"
        assert result.attrs[SPACE_TRAIT_ATTR] == "grid"
        assert result.attrs[UNCERTAINTY_TRAIT_ATTR] == "deterministic"

    def test_dims_after_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        preprocessed = anemoi_inference_xy._preprocess(_make_raw_dataset())

        monkeypatch.setattr(
            anemoi_inference_xy.xr,
            "open_mfdataset",
            lambda *a, **kw: preprocessed,
        )

        result = anemoi_inference_xy.load_dataset("dummy.nc")
        assert "reference_time" in result.dims
        assert "lead_time" in result.dims
        assert "y" in result.dims
        assert "x" in result.dims
        assert "time" not in result.dims

    def test_multi_file_paths_passed_to_open_mfdataset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Multiple file paths are forwarded as a list to xr.open_mfdataset."""
        captured: dict[str, object] = {}
        preprocessed = anemoi_inference_xy._preprocess(_make_raw_dataset())

        def _fake_open(paths, **kw):
            captured["paths"] = paths
            return preprocessed

        monkeypatch.setattr(anemoi_inference_xy.xr, "open_mfdataset", _fake_open)

        anemoi_inference_xy.load_dataset(["a.nc", "b.nc"])
        assert captured["paths"] == ["a.nc", "b.nc"]

    def test_single_string_path_wrapped_in_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A single string path is converted to a one-element list."""
        captured: dict[str, object] = {}
        preprocessed = anemoi_inference_xy._preprocess(_make_raw_dataset())

        def _fake_open(paths, **kw):
            captured["paths"] = paths
            return preprocessed

        monkeypatch.setattr(anemoi_inference_xy.xr, "open_mfdataset", _fake_open)

        anemoi_inference_xy.load_dataset("single.nc")
        assert captured["paths"] == ["single.nc"]

    def test_weather_variables_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        preprocessed = anemoi_inference_xy._preprocess(_make_raw_dataset())

        monkeypatch.setattr(
            anemoi_inference_xy.xr,
            "open_mfdataset",
            lambda *a, **kw: preprocessed,
        )

        result = anemoi_inference_xy.load_dataset("dummy.nc")
        assert "2t" in result.data_vars
        assert "tp" in result.data_vars

    @pytest.mark.parametrize("ny,nx", [(3, 4), (5, 5)])
    def test_spatial_shape_preserved(
        self, monkeypatch: pytest.MonkeyPatch, ny: int, nx: int
    ) -> None:
        preprocessed = anemoi_inference_xy._preprocess(_make_raw_dataset(ny=ny, nx=nx))

        monkeypatch.setattr(
            anemoi_inference_xy.xr,
            "open_mfdataset",
            lambda *a, **kw: preprocessed,
        )

        result = anemoi_inference_xy.load_dataset("dummy.nc")
        assert result.sizes["y"] == ny
        assert result.sizes["x"] == nx
