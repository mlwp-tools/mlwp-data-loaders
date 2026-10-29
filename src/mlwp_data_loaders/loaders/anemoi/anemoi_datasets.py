"""
Loader for anemoi-datasets Zarr stores.

The store holds a single ``data`` array ``(time, variable, ensemble, cell)``
whose chunks typically contain one time step of *all* variables. To keep the
dask graph of the returned dataset small, independent of the length of the
record and of the number of variables, this loader

1. opens the store without dask, so selections on the backend array are free,
2. creates one dask array with one chunk per on-disk time chunk, read with
   basic (slice/integer) indexing only, which lets zarr decode each chunk
   straight into the output buffer,
3. splits the variables with one non-materialized ``Blockwise`` layer each,
   all reading from that shared array, so a compute reads every on-disk chunk
   once no matter how many variables it needs.
"""

from typing import Any

import dask.array
import numpy as np
import pandas as pd
import xarray as xr
from mlwp_data_specs.api import (
    SPACE_TRAIT_ATTR,
    TIME_TRAIT_ATTR,
    UNCERTAINTY_TRAIT_ATTR,
    Space,
    Time,
    Uncertainty,
)

DROP_VARS = [
    "latitude",
    "longitude",
    "time",
    "cos_julian_day",
    "cos_latitude",
    "cos_local_time",
    "cos_longitude",
    "insolation",
    "sin_julian_day",
    "sin_latitude",
    "sin_local_time",
    "sin_longitude",
]


def load_dataset(
    path: str,
    variables: str | list[str] | None = None,
    valid_times: slice | None = None,
    consolidated: bool = False,
    storage_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> xr.Dataset:
    """
    Load an anemoi-datasets Zarr store as a lazy, dask-backed dataset.

    Each variable is chunked as one dask chunk per on-disk time chunk (one time
    step for anemoi-datasets stores) with ``grid_index`` in a single chunk.
    Larger time chunks are not offered: every chunk holds all variables of the
    store, so they would multiply the memory and IO of each read.

    Parameters
    ----------
    path : str
        Path to the Zarr dataset.
    variables : str or list of str, optional
        Variables to load. Defaults to all variables except forcings and
        constant fields (``DROP_VARS``).
    valid_times : slice, optional
        Label slice of valid times to load, inclusive at both ends like
        ``Dataset.sel`` (e.g. ``slice("2020-01-01", "2020-12-31")``). Defaults
        to the full record. When computing many variables at once, dask's
        graph optimization scales with ``n_variables * n_time_steps`` of the
        loaded record, so subsetting here keeps that cheap.
    consolidated : bool, default: False
        Whether to use consolidated metadata when opening the Zarr store.
    storage_options : dict of str to Any, optional
        Storage options passed to xarray.open_zarr (e.g. for S3 access).
    **kwargs
        Additional keyword arguments passed to xarray.open_zarr.

    Returns
    -------
    xr.Dataset
        ``(valid_time, grid_index)`` variables with latitude/longitude
        coordinates and the mlwp trait attributes.

    Raises
    ------
    ValueError
        If any of ``variables`` is not in the dataset, or no time steps fall
        within ``valid_times``.
    """
    # No dask yet: indexing the backend arrays below only composes an indexer
    ds = xr.open_zarr(
        path,
        chunks=None,
        consolidated=consolidated,
        storage_options=storage_options,
        **kwargs,
    )  # type: ignore

    # all variables in the store, in the order of its `variable` axis
    available_variable_names = list(ds.attrs["variables"])
    if variables is None:
        variables = [name for name in available_variable_names if name not in DROP_VARS]
    elif isinstance(variables, str):
        variables = [variables]
    missing = [name for name in variables if name not in available_variable_names]
    if missing:
        raise ValueError(f"Variables {missing} not found in {path}")

    dates = pd.DatetimeIndex(ds["dates"].values.astype("datetime64[ns]"))
    time_slice = slice(None)
    if valid_times is not None:
        time_slice = dates.slice_indexer(valid_times.start, valid_times.stop)
        if len(dates[time_slice]) == 0:
            raise ValueError(f"No time steps of {path} within {valid_times}")

    # Create one lazy (time, variable, cell) dask array that all the
    # per-variable arrays below are taken from, so computing several variables
    # reads and decodes each on-disk chunk only once rather than once per
    # variable. Each dask chunk is exactly one on-disk chunk: smaller would
    # decode a zarr chunk several times, larger would hold several time steps
    # of all variables in memory per task.
    # All variables are kept, since an on-disk chunk holds all of them and
    # selecting some would read no less; indexing `variable` with a list (or
    # `ensemble` with anything but an integer) would also switch every read to
    # zarr's orthogonal indexing, about 3x slower and needing 2x the memory.
    store_data: dask.array.Array = (
        ds["data"]
        .isel(time=time_slice, ensemble=0)
        .chunk(
            {
                "time": ds["data"].encoding["chunks"][0],  # on-disk chunk
                "variable": -1,  # all variables in one chunk
                "cell": -1,  # the whole grid in one chunk
            }
        )
        .data
    )
    # Split `store_data` into one lazy (valid_time, grid_index) array per
    # variable. Each is a single dask layer that calls _extract_variable_array
    # on every chunk of `store_data`, so computing several variables still
    # reads each on-disk chunk only once.
    # `empty_result` is an empty array of the result's type, dtype and number
    # of dims, so dask need not call _extract_variable_array to find them out
    empty_result: np.ndarray = np.empty((0, 0), dtype=store_data.dtype)
    data_vars: dict[str, tuple[tuple[str, str], dask.array.Array]] = {}
    for name in variables:
        # position of the variable along the store's `variable` axis
        index = available_variable_names.index(name)
        variable_data: dask.array.Array = store_data.map_blocks(
            _extract_variable_array,
            index,
            drop_axis=1,  # the variable axis is not in the result
            meta=empty_result,
            token=f"anemoi-{name}",  # stable, readable name for the layer
        )
        data_vars[name] = (("valid_time", "grid_index"), variable_data)

    coords = {
        "valid_time": (
            "valid_time",
            dates[time_slice].values,
            {"standard_name": "time"},
        ),
        "latitude": (
            "grid_index",
            ds["latitudes"].values.astype(np.float32),
            {"standard_name": "latitude", "units": "degrees_north"},
        ),
        "longitude": (
            "grid_index",
            ds["longitudes"].values.astype(np.float32),
            {"standard_name": "longitude", "units": "degrees_east"},
        ),
    }

    ds_out = xr.Dataset(data_vars, coords=coords)
    ds_out.attrs[TIME_TRAIT_ATTR] = Time.OBSERVATION
    ds_out.attrs[SPACE_TRAIT_ATTR] = Space.GRID
    ds_out.attrs[UNCERTAINTY_TRAIT_ATTR] = Uncertainty.DETERMINISTIC
    return ds_out


def _extract_variable_array(array: np.ndarray, index: int) -> np.ndarray:
    """Extract one variable from a ``(time, variable, cell)`` array.

    ``map_blocks`` calls this once per chunk of the dask array, with that
    chunk's values as a numpy array (what dask calls a block): one on-disk
    time chunk (usually one time step) of all variables on the whole grid.

    The copy lets the array holding all variables be freed as soon as the
    variables needed from it are extracted, instead of living as long as a
    view.

    Parameters
    ----------
    array : np.ndarray
        ``(time, variable, cell)`` values of one chunk.
    index : int
        Position of the variable along the ``variable`` axis.

    Returns
    -------
    np.ndarray
        ``(time, cell)`` copy of the variable's values.
    """
    return array[:, index].copy()
