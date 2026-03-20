"""Loader for reading HARP SQLite observation tables."""

from __future__ import annotations

import sqlite3

import pandas as pd
import xarray as xr

TIME_PROFILE = "observation"
SPACE_PROFILE = "point"
UNCERTAINTY_PROFILE = "deterministic"

COORDS = {
    "longitude": "lon",
    "latitude": "lat",
    "valid_time": "validdate",
    "code": "SID",
    "altitude": "elev",
}


def load_dataset(
    paths: str | list[str], variables: list[str] | None = None, **kwargs
) -> xr.Dataset:
    """
    Load HARP observation datasets from SQLite files.

    Parameters
    ----------
    paths : str | list[str]
        Path to the SQLite file. Currently only supports a single path.
    variables : list[str] | None, optional
        List of variables to select. If None, all variables except coordinates are read.
    **kwargs
        Additional arguments (currently ignored, provided for compatibility).

    Returns
    -------
    xr.Dataset
        The loaded xarray Dataset.
    """
    if isinstance(paths, list):
        if len(paths) > 1:
            raise NotImplementedError(
                "Reading from multiple SQLite-files not implemented"
            )
        path = paths[0]
    else:
        path = paths

    # Connect to the sqlite file
    conn = sqlite3.connect(path)

    if variables is None:
        # Retrieve all variables by checking the table columns and filtering out known coords
        variables_query = pd.read_sql_query("SELECT * FROM SYNOP LIMIT 0", conn)
        variables = [
            var for var in variables_query.columns if var not in COORDS.values()
        ]

    # Read the station details (coordinates) using SID as code
    codes = pd.read_sql(
        "SELECT SID as code, MIN(lat) AS latitude, MIN(lon) AS longitude, elev as altitude FROM SYNOP GROUP BY SID",
        conn,
        index_col="code",
    ).to_xarray()

    # Read the data
    query = f"""
        SELECT SID as code, validdate as valid_time, {", ".join(variables)}
        FROM SYNOP
    """

    df = pd.read_sql(
        query,
        conn,
        index_col=["code", "valid_time"],
        parse_dates={"valid_time": {"unit": "s"}},
    )

    ds = df.to_xarray()

    # Align coordinates using the code index
    lon_values = codes["longitude"].sel(code=ds["code"]).values
    lat_values = codes["latitude"].sel(code=ds["code"]).values
    alt_values = codes["altitude"].sel(code=ds["code"]).values

    ds = ds.assign_coords(
        longitude=("code", lon_values),
        latitude=("code", lat_values),
        altitude=("code", alt_values),
    )

    ds.coords["valid_time"].attrs["standard_name"] = "time"
    ds.coords["latitude"].attrs.update(
        {"standard_name": "latitude", "units": "degrees_north"}
    )
    ds.coords["longitude"].attrs.update(
        {"standard_name": "longitude", "units": "degrees_east"}
    )

    return ds.rename_dims({"code": "point_index"}).transpose(
        "valid_time", "point_index"
    )
