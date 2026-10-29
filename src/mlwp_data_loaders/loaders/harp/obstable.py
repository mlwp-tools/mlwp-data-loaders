"""
Loader for reading HARP SQLite observation tables.

The ``SYNOP`` table holds one row per station and valid time, indexed by
``(validdate, SID)``. To keep loading cheap and the dask graph of the returned
dataset small, independent of the length of the record and of the number of
variables, this loader

1. reads only the valid times and the coordinates of every station when
   loading,
2. creates one dask array with one chunk per range of valid times, each task
   reading all requested variables of its range with one indexed query,
3. splits the variables with one non-materialized ``Blockwise`` layer each,
   all reading from that shared array, so a compute reads every row once no
   matter how many variables it needs.

Reading costs about 6 µs per row, almost all of it spent by ``sqlite3``
building Python objects, so one query for all variables is much cheaper than
one query per variable.
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import xarray as xr
from dask.utils import parse_bytes
from mlwp_data_specs.api import (
    SPACE_TRAIT_ATTR,
    TIME_TRAIT_ATTR,
    UNCERTAINTY_TRAIT_ATTR,
    Space,
    Time,
    Uncertainty,
)

TABLE = "SYNOP"
COORDS = {
    "longitude": "lon",
    "latitude": "lat",
    "valid_time": "validdate",
    "code": "SID",
    "altitude": "elev",
}
# Each station's coordinates are those of its latest row, so that they come
# from one row even for stations whose coordinates change in the table
STATIONS_QUERY = f"""
    SELECT s.SID, s.lat, s.lon, s.elev
    FROM (SELECT SID, MAX(validdate) AS validdate FROM {TABLE} GROUP BY SID) AS latest
    JOIN {TABLE} AS s ON s.validdate = latest.validdate AND s.SID = latest.SID
    ORDER BY s.SID
"""
VALID_TIMES_QUERY = f"SELECT DISTINCT validdate FROM {TABLE} ORDER BY validdate"
# Rows converted to numpy at a time, so that the Python objects sqlite3 builds
# for them stay small next to the block they are written into
FETCH_ROWS = 65536
# sqlite3 releases the GIL around every row it fetches, so reads running in
# parallel threads (dask's default scheduler) take turns on the GIL for every
# row and run 10-40x slower than one after the other. The rows are Python
# objects built under the GIL anyway, so serializing the reads costs little.
_READ_LOCK = threading.Lock()


def load_dataset(
    paths: str | list[str], variables: str | list[str] | None = None, **kwargs
) -> xr.Dataset:
    """
    Load a HARP SQLite observation table as a lazy, dask-backed dataset.

    Each dask chunk holds a range of valid times of all stations, with as
    many valid times as fit in dask's ``array.chunk-size`` (128 MiB by
    default) for all of ``variables`` together, since one task reads them
    all. Set it with ``dask.config.set({"array.chunk-size": ...})``. Large
    chunks keep the graph small, also after aligning the observations to
    forecasts.

    Each station's latitude, longitude and altitude are taken from its latest
    row in the table.

    Parameters
    ----------
    paths : str | list[str]
        Path to the SQLite file. Currently only supports a single path.
    variables : str or list of str, optional
        Columns of the ``SYNOP`` table to load. Defaults to all columns except
        the coordinates.
    **kwargs
        Additional arguments (currently ignored, provided for compatibility).

    Returns
    -------
    xr.Dataset
        ``(valid_time, point_index)`` variables with ``code``, latitude,
        longitude and altitude coordinates and the mlwp trait attributes.

    Raises
    ------
    NotImplementedError
        If more than one path is given.
    FileNotFoundError
        If the SQLite file does not exist.
    ValueError
        If the file has no observation table, the table is empty, or any of
        ``variables`` is not a column of it.
    """
    if isinstance(paths, list):
        if len(paths) > 1:
            raise NotImplementedError(
                "Reading from multiple SQLite-files not implemented"
            )
        path = paths[0]
    else:
        path = paths
    if not Path(path).is_file():
        raise FileNotFoundError(f"No such SQLite file: {path!r}")
    if isinstance(variables, str):
        variables = [variables]

    with _connect(path) as conn:
        columns = [row[1] for row in conn.execute(f"PRAGMA table_info({TABLE})")]
        if not columns:
            raise ValueError(f"No {TABLE} table in {path}")
        names = [name for name in columns if name not in COORDS.values()]
        if variables is None:
            variables = names
        missing = [name for name in variables if name not in names]
        if missing:
            raise ValueError(
                f"Variables {missing} not found in the {TABLE} table of {path}"
            )
        times = np.array(
            [time for (time,) in conn.execute(VALID_TIMES_QUERY)], dtype=np.int64
        )
        stations = np.array(conn.execute(STATIONS_QUERY).fetchall(), dtype=np.float64)
    if times.size == 0:
        raise ValueError(f"The {TABLE} table of {path} is empty")
    codes = stations[:, 0].astype(np.int64)

    # One chunk per range of valid times; each read task gets a block holding
    # the path and the valid times of its chunk
    chunks = _time_chunks(times.size, len(variables) * codes.size)
    bounds = np.cumsum((0, *chunks))
    blocks = np.empty(len(chunks), dtype=object)
    for i, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:])):
        blocks[i] = (path, times[start:stop])
    data = da.from_array(blocks, chunks=1).map_blocks(
        _read_block,
        variables,
        codes,
        new_axis=[1, 2],
        chunks=(chunks, (len(variables),), (codes.size,)),
        meta=np.empty((0, 0, 0), dtype=np.float64),
        token="harp-obstable-read",
    )
    meta = np.empty((0, 0), dtype=np.float64)
    data_vars = {
        name: (
            ("valid_time", "code"),
            data.map_blocks(
                _take_variable,
                index,
                drop_axis=1,
                meta=meta,
                token=f"harp-obstable-{name}",
            ),
        )
        for index, name in enumerate(variables)
    }
    coords = {
        "valid_time": times.astype("datetime64[s]").astype("datetime64[ns]"),
        "code": codes,
        "longitude": ("code", stations[:, 2]),
        "latitude": ("code", stations[:, 1]),
        "altitude": ("code", stations[:, 3]),
    }
    ds = xr.Dataset(data_vars, coords=coords)

    ds.coords["valid_time"].attrs["standard_name"] = "time"
    ds.coords["latitude"].attrs.update(
        {"standard_name": "latitude", "units": "degrees_north"}
    )
    ds.coords["longitude"].attrs.update(
        {"standard_name": "longitude", "units": "degrees_east"}
    )

    ds_final = ds.rename_dims({"code": "point_index"})

    ds_final.attrs[TIME_TRAIT_ATTR] = Time.OBSERVATION
    ds_final.attrs[SPACE_TRAIT_ATTR] = Space.POINT
    ds_final.attrs[UNCERTAINTY_TRAIT_ATTR] = Uncertainty.DETERMINISTIC

    return ds_final


@contextlib.contextmanager
def _connect(path: str) -> Iterator[sqlite3.Connection]:
    """Open the SQLite file ``path`` read-only and close it afterwards.

    ``sqlite3.connect(path)`` would create an empty database for a missing
    file, and its context manager only ends a transaction, it doesn't close.

    Parameters
    ----------
    path : str
        Path to the SQLite file.

    Yields
    ------
    sqlite3.Connection
        Read-only connection to the file.
    """
    conn = sqlite3.connect(f"{Path(path).absolute().as_uri()}?mode=ro", uri=True)
    try:
        yield conn
    finally:
        conn.close()


def _time_chunks(n_times: int, values_per_time: int) -> tuple[int, ...]:
    """Split ``n_times`` valid times into equal chunks of dask's ``array.chunk-size``.

    Parameters
    ----------
    n_times : int
        Number of valid times to split.
    values_per_time : int
        Number of values one valid time holds: all variables at all stations.

    Returns
    -------
    tuple of int
        Number of valid times in each chunk, differing by at most one.
    """
    limit = parse_bytes(dask.config.get("array.chunk-size"))
    per_chunk = max(1, limit // (8 * max(1, values_per_time)))
    n_chunks = -(-n_times // per_chunk)
    size, extra = divmod(n_times, n_chunks)
    return (size + 1,) * extra + (size,) * (n_chunks - extra)


def _quote(name: str) -> str:
    """Quote ``name`` as an SQL identifier.

    Parameters
    ----------
    name : str
        Table or column name.

    Returns
    -------
    str
        ``name`` in double quotes, with any double quotes in it doubled.
    """
    return '"' + name.replace('"', '""') + '"'


def _read_block(
    block: np.ndarray, variables: list[str], codes: np.ndarray
) -> np.ndarray:
    """Read one chunk of valid times as a ``(valid_time, variable, station)`` block.

    Values missing from the table, either NULL or without a row for that
    station and valid time, are NaN.

    Parameters
    ----------
    block : np.ndarray
        Object array holding one ``(path, times)`` pair: the SQLite file and
        the sorted valid times (as stored in ``validdate``) of the chunk.
    variables : list of str
        Columns of the observation table to read.
    codes : np.ndarray
        Sorted station codes (``SID``), giving the order of the station axis.

    Returns
    -------
    np.ndarray
        ``(valid_time, variable, station)`` float64 block.
    """
    path, times = block[0]
    columns = ", ".join(_quote(name) for name in ["validdate", "SID", *variables])
    query = f"SELECT {columns} FROM {TABLE} WHERE validdate BETWEEN ? AND ?"
    out = np.full((times.size, len(variables), codes.size), np.nan)
    with _connect(path) as conn, _READ_LOCK:
        cursor = conn.execute(query, (int(times[0]), int(times[-1])))
        while rows := cursor.fetchmany(FETCH_ROWS):
            values = np.array(rows, dtype=np.float64)
            time_index = np.searchsorted(times, values[:, 0].astype(np.int64))
            station_index = np.searchsorted(codes, values[:, 1].astype(np.int64))
            out[time_index, :, station_index] = values[:, 2:]
    return out


def _take_variable(block: np.ndarray, index: int) -> np.ndarray:
    """Return one variable of a ``(valid_time, variable, station)`` block.

    The copy lets the block holding all variables be freed as soon as the
    variables needed from it are taken, instead of living as long as a view.

    Parameters
    ----------
    block : np.ndarray
        ``(valid_time, variable, station)`` block holding all variables.
    index : int
        Position of the variable along the ``variable`` axis.

    Returns
    -------
    np.ndarray
        ``(valid_time, station)`` copy of the variable's values.
    """
    return block[:, index].copy()
