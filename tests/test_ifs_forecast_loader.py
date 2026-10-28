"""Tests for the IFS forecast loader on small GRIB files written with eccodes."""

from __future__ import annotations

import dask
import dask.array
import numpy as np
import pytest
from loguru import logger

from mlwp_data_loaders.api import load_and_validate_dataset
from mlwp_data_loaders.ensemble import MEMBER_ATTRS

eccodes = pytest.importorskip("eccodes")
cfgrib = pytest.importorskip("cfgrib")

from mlwp_data_loaders.loaders.ifs import forecast  # noqa: E402

LOADER = "mlwp_data_loaders.loaders.ifs.forecast"
SAMPLE = "reduced_gg_pl_32_grib1"
N_GRID = 6114
HOUR = np.timedelta64(1, "h")


def _value(date: int, number: int, step: int) -> float:
    """Return the constant value written to the field of ``date``, ``number`` and ``step``."""
    return 100.0 * number + step + date % 100


def _write_grib(
    path,
    data_type: str = "fc",
    members: tuple[int, ...] = (0,),
    steps: tuple[int, ...] = (0, 6, 12),
    dates: tuple[int, ...] = (20260101,),
    short_names: tuple[str, ...] = ("2t", "10u"),
) -> str:
    """Write one surface field per date, variable, member and step to ``path``."""
    with open(path, "wb") as f:
        for date in dates:
            for short_name in short_names:
                for number in members:
                    for step in steps:
                        h = eccodes.codes_grib_new_from_samples(SAMPLE)
                        eccodes.codes_set(h, "shortName", short_name)
                        eccodes.codes_set(h, "typeOfLevel", "surface")
                        eccodes.codes_set(
                            h, "stream", "oper" if data_type == "fc" else "enfo"
                        )
                        eccodes.codes_set(h, "type", data_type)
                        if data_type != "fc":
                            eccodes.codes_set(h, "number", number)
                        eccodes.codes_set(h, "dataDate", date)
                        eccodes.codes_set(h, "dataTime", 0)
                        eccodes.codes_set(h, "step", step)
                        values = np.full(N_GRID, _value(date, number, step))
                        eccodes.codes_set_values(h, values)
                        eccodes.codes_write(h, f)
                        eccodes.codes_release(h)
    return str(path)


@pytest.fixture
def warnings_logged():
    """Collect the messages of loguru warnings logged during a test."""
    messages = []
    handler_id = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    yield messages
    logger.remove(handler_id)


@pytest.fixture
def field_reads(monkeypatch: pytest.MonkeyPatch) -> list:
    """Count the GRIB fields cfgrib decodes."""
    reads = []
    get_values_in_order = cfgrib.dataset.get_values_in_order

    def counting(*args, **kwargs):
        reads.append(1)
        return get_values_in_order(*args, **kwargs)

    monkeypatch.setattr(cfgrib.dataset, "get_values_in_order", counting)
    return reads


def _load_and_validate(paths: list[str], **kwargs):
    """Load ``paths`` with the IFS loader and check the validation passes."""
    ds, report = load_and_validate_dataset(
        paths, loader=LOADER, return_validation_report=True, **kwargs
    )
    if report.has_fails():
        report.console_print()
    assert not report.has_fails()
    return ds


def test_deterministic(tmp_path) -> None:
    """HRES files give lazy ``(reference_time, lead_time, grid_index)`` variables."""
    paths = [
        _write_grib(tmp_path / f"fc_{date}.grib", dates=(date,))
        for date in (20260101, 20260102)
    ]
    ds = _load_and_validate(paths)

    assert ds["t2m"].dims == ("reference_time", "lead_time", "grid_index")
    assert "member" not in ds.coords
    assert ds.attrs["mlwp_uncertainty_trait"] == "deterministic"
    assert isinstance(ds["t2m"].data, dask.array.Array)
    assert ds["t2m"].data.chunksize == (1, 1, N_GRID)
    np.testing.assert_array_equal(ds["lead_time"].values, np.array([0, 6, 12]) * HOUR)
    assert ds["longitude"].min() >= -180 and ds["longitude"].max() < 180
    np.testing.assert_array_equal(
        ds["t2m"].isel(grid_index=0).values,
        [[_value(date, 0, step) for step in (0, 6, 12)] for date in (1, 2)],
    )


def test_ensemble(tmp_path) -> None:
    """Perturbed members give a leading integer ``member`` dim with the CF attrs."""
    paths = [
        _write_grib(tmp_path / f"pf_{date}.grib", "pf", (1, 2, 3), dates=(date,))
        for date in (20260101, 20260102)
    ]
    ds = _load_and_validate(paths)

    assert ds["t2m"].dims == ("member", "reference_time", "lead_time", "grid_index")
    assert ds["member"].values.tolist() == [1, 2, 3]
    assert ds["member"].dtype == np.int64
    assert ds["member"].attrs == MEMBER_ATTRS
    assert ds.attrs["mlwp_uncertainty_trait"] == "ensemble"
    assert ds["t2m"].data.chunksize == (3, 1, 1, N_GRID)
    value = ds["u10"].sel(member=2, lead_time=6 * HOUR).isel(grid_index=0).values
    np.testing.assert_array_equal(value, [_value(1, 2, 6), _value(2, 2, 6)])


def test_single_member_is_scalar(tmp_path) -> None:
    """A control-forecast file is deterministic with a scalar ``member``."""
    paths = [_write_grib(tmp_path / "cf.grib", "cf", (0,))]
    ds = _load_and_validate(paths)

    assert ds.attrs["mlwp_uncertainty_trait"] == "deterministic"
    assert ds["member"].dims == ()
    assert int(ds["member"]) == 0
    assert ds["member"].attrs == MEMBER_ATTRS


def test_members_split_up_to_chunk_size(tmp_path) -> None:
    """Members share a chunk only as far as they fit in ``array.chunk-size``."""
    paths = [_write_grib(tmp_path / "pf.grib", "pf", (1, 2, 3))]
    # A field is 6114 float32 values (24 kB), so two fit in 60 kB
    with dask.config.set({"array.chunk-size": "60kB"}):
        ds = forecast.load_dataset(paths)

    assert ds["t2m"].chunks[0] == (2, 1)


def test_reads_only_selected_fields(tmp_path, field_reads) -> None:
    """Loading reads no fields, and selecting a lead time reads only its fields."""
    paths = [
        _write_grib(tmp_path / f"pf_{date}.grib", "pf", (1, 2, 3), dates=(date,))
        for date in (20260101, 20260102)
    ]
    ds = forecast.load_dataset(paths, variables="t2m")
    assert len(field_reads) == 0

    ds["t2m"].isel(lead_time=1).compute()
    assert len(field_reads) == 2 * 3  # reference times x members


def test_several_reference_times_per_file(tmp_path) -> None:
    """A file may hold several forecasts; a single lead time stays a dim."""
    paths = [_write_grib(tmp_path / "fc.grib", dates=(20260101, 20260102), steps=(6,))]
    ds = _load_and_validate(paths)

    assert dict(ds["t2m"].sizes) == {
        "reference_time": 2,
        "lead_time": 1,
        "grid_index": N_GRID,
    }
    assert ds["t2m"].data.chunksize == (1, 1, N_GRID)


def test_skips_file_with_other_lead_times(tmp_path, warnings_logged) -> None:
    """A file with fewer lead times (a run that stopped early) is skipped."""
    paths = [
        _write_grib(tmp_path / "fc_1.grib", dates=(20260101,)),
        _write_grib(tmp_path / "fc_2.grib", dates=(20260102,), steps=(0,)),
        _write_grib(tmp_path / "fc_3.grib", dates=(20260103,)),
    ]
    ds = forecast.load_dataset(paths)

    assert ds.sizes["reference_time"] == 2
    assert ds.sizes["lead_time"] == 3
    assert len(warnings_logged) == 1
    assert "fc_2.grib" in warnings_logged[0]


def test_missing_variable_raises(tmp_path) -> None:
    """Asking for a variable that is not in the files raises."""
    paths = [_write_grib(tmp_path / "fc.grib")]
    with pytest.raises(ValueError, match="msl"):
        forecast.load_dataset(paths, variables=["t2m", "msl"])


def test_chunks_argument_raises(tmp_path) -> None:
    """The removed ``chunks`` argument raises instead of being passed to xarray."""
    paths = [_write_grib(tmp_path / "fc.grib")]
    with pytest.raises(TypeError, match="chunks"):
        forecast.load_dataset(paths, chunks={"step": 1})
