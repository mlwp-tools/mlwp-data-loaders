"""Tests for the HARP obstable loader on small SQLite files."""

from __future__ import annotations

import sqlite3

import dask
import numpy as np
import pandas as pd
import pytest

from mlwp_data_loaders.api import load_and_validate_dataset
from mlwp_data_loaders.loaders.harp.obstable import load_dataset

LOADER = "mlwp_data_loaders.loaders.harp.obstable"
TIMES = pd.date_range("2026-01-01", periods=6, freq="h")


def _rows() -> pd.DataFrame:
    """Rows of a SYNOP table with three stations and six valid times.

    Station 6451 has no row at the second time, station 6400 reports only at
    the first time, and station 6447 moves before its last report.
    """
    stations = pd.DataFrame(
        {
            "SID": [6447, 6451],
            "lat": [50.8, 51.2],
            "lon": [4.4, 2.9],
            "elev": [100.0, 4.0],
        }
    )
    # HARP stores valid times as unix seconds
    validdate = pd.DataFrame({"validdate": TIMES.as_unit("s").asi8})
    rows = stations.merge(validdate, how="cross")
    rows = rows.drop(index=rows.index[(rows["SID"] == 6451)][1])
    extra = {"SID": 6400, "lat": 49.6, "lon": 5.8, "elev": 300.0}
    rows = pd.concat(
        [rows, pd.DataFrame([{**extra, "validdate": validdate["validdate"][0]}])]
    )
    moved = (rows["SID"] == 6447) & (rows["validdate"] == validdate["validdate"].max())
    rows.loc[moved, ["lat", "lon", "elev"]] = [50.9, 4.5, 101.0]
    rows = rows.sort_values(["validdate", "SID"], ignore_index=True)
    rows["T2m"] = np.arange(len(rows), dtype=float) + 270.0
    rows.loc[3, "T2m"] = np.nan  # NULL in the table
    # Needs quoting: unquoted, SQL reads it as the number 10 named si
    rows["10si"] = np.arange(len(rows), dtype=float)
    return rows


@pytest.fixture
def rows() -> pd.DataFrame:
    """Return the observation rows written to the test table."""
    return _rows()


@pytest.fixture
def obstable(tmp_path, rows) -> str:
    """Write the rows as a SYNOP table with HARP's index."""
    path = str(tmp_path / "OBSTABLE.sqlite")
    with sqlite3.connect(path) as conn:
        rows.to_sql("SYNOP", conn, index=False)
        conn.execute("CREATE UNIQUE INDEX index_validdate_SID ON SYNOP(validdate,SID)")
    return path


def _expected(rows: pd.DataFrame, name: str) -> np.ndarray:
    """Return ``name`` as a dense ``(valid_time, station)`` array, NaN where missing."""
    table = rows.pivot(index="validdate", columns="SID", values=name)
    return table.reindex(columns=sorted(rows["SID"].unique())).to_numpy()


def test_load_dataset_matches_the_table(obstable, rows) -> None:
    """Every station and valid time is there, NaN where the table has no value."""
    ds = load_dataset(obstable)

    assert list(ds.data_vars) == ["T2m", "10si"]
    assert ds["T2m"].dims == ("valid_time", "point_index")
    assert ds["code"].values.tolist() == [6400, 6447, 6451]
    np.testing.assert_array_equal(ds["valid_time"].values, TIMES.values)
    assert ds["valid_time"].dtype == np.dtype("datetime64[ns]")
    for name in ds.data_vars:
        np.testing.assert_array_equal(ds[name].values, _expected(rows, name))


def test_load_dataset_is_lazy(obstable, monkeypatch) -> None:
    """Loading reads no rows of data; computing reads each chunk with one query."""
    statements = []
    connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        conn = connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(sqlite3, "connect", traced_connect)
    # 48 bytes per valid time (2 variables, 3 stations): 4 fit, split evenly
    with dask.config.set({"array.chunk-size": "200B"}):
        ds = load_dataset(obstable)

    def reads():
        return [s for s in statements if "BETWEEN" in s]

    assert reads() == []
    assert ds["T2m"].chunks == ((3, 3), (3,))
    graph = ds.__dask_graph__()
    # chunk descriptors, the shared read, and one layer per variable
    assert len(graph.layers) == 2 + len(ds.data_vars)

    ds.compute()
    assert len(reads()) == 2


def test_load_dataset_chunks_hold_the_same_values(obstable, rows) -> None:
    """Splitting the valid times into chunks doesn't change the values."""
    with dask.config.set({"array.chunk-size": "100B"}):
        ds = load_dataset(obstable)
    assert ds["T2m"].chunks == ((2, 2, 2), (3,))
    np.testing.assert_array_equal(ds["T2m"].values, _expected(rows, "T2m"))


def test_threads_give_the_same_values_as_sync(obstable) -> None:
    """Reads serialized by the module lock still give the right blocks."""
    with dask.config.set({"array.chunk-size": "100B"}):
        ds = load_dataset(obstable)
    threaded = ds.compute(scheduler="threads", num_workers=4)
    sync = ds.compute(scheduler="sync")
    for name in ds.data_vars:
        np.testing.assert_array_equal(threaded[name].values, sync[name].values)


def test_station_coordinates_come_from_the_latest_row(obstable) -> None:
    """Station 6447 moved before its last report; its last position is used."""
    ds = load_dataset(obstable)

    station = ds.sel(code=6447)
    assert station["latitude"].item() == 50.9
    assert station["longitude"].item() == 4.5
    assert station["altitude"].item() == 101.0
    assert ds["latitude"].attrs["standard_name"] == "latitude"


def test_load_dataset_selects_variables(obstable, rows) -> None:
    """A variable can be given as a string, and names needing quotes are read."""
    ds = load_dataset(obstable, variables="10si")

    assert list(ds.data_vars) == ["10si"]
    np.testing.assert_array_equal(ds["10si"].values, _expected(rows, "10si"))


def test_load_dataset_raises_on_unknown_variable(obstable) -> None:
    """Unknown variables are reported, matched case-sensitively."""
    with pytest.raises(ValueError, match=r"\['t2m'\] not found"):
        load_dataset(obstable, variables=["T2m", "t2m"])


def test_load_dataset_raises_on_missing_file(tmp_path) -> None:
    """A missing file raises instead of being created as an empty database."""
    path = tmp_path / "OBSTABLE_missing.sqlite"
    with pytest.raises(FileNotFoundError):
        load_dataset(str(path))
    assert not path.exists()


def test_load_and_validate_dataset_passes_validation(obstable) -> None:
    """The loaded dataset passes the mlwp-data-specs validation."""
    _, report = load_and_validate_dataset(
        obstable, loader=LOADER, return_validation_report=True
    )
    if report.has_fails():
        report.console_print()
    assert not report.has_fails()
