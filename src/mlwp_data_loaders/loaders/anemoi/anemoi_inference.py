"""
Loader for anemoi-inference NetCDF output.

Every file holds one forecast: ``(time, values)`` variables, where ``time`` is
the unlimited dimension of valid times and ``values`` the grid points. To keep
loading cheap and the dask graph of the returned dataset independent of the
number of files, this loader

1. reads only the ``time`` variable and the variable names of every file, and
   the grid and variable attributes of one reference file,
2. builds every variable as one non-materialized ``Blockwise`` layer with one
   chunk per file, each task reading that variable from one file.

Files are read with h5netcdf directly rather than with ``xarray.open_dataset``.
Its h5netcdf backend looks up the shape of every variable, and the size of an
unlimited dimension is found by inspecting every variable on it, so opening a
file costs O(n_variables**2): about 3 s per file with 70 variables.
"""

import contextlib
from collections import Counter
from collections.abc import Iterator
from typing import Any

import dask.array as da
import fsspec
import h5netcdf
import numpy as np
import xarray as xr
from loguru import logger
from mlwp_data_specs.api import (
    SPACE_TRAIT_ATTR,
    TIME_TRAIT_ATTR,
    UNCERTAINTY_TRAIT_ATTR,
    Space,
    Time,
    Uncertainty,
)

DIMS = ("time", "values")


def load_dataset(
    paths: str | list[str],
    variables: str | list[str] | None = None,
    storage_options: dict[str, Any] | None = None,
) -> xr.Dataset:
    """
    Load anemoi-inference NetCDF files as one lazy, dask-backed dataset.

    Every file is one forecast and becomes one ``reference_time``: the first
    valid time in that file. Each variable has one dask chunk per file, holding
    all its lead times and grid points.

    Files whose lead times differ from the most common ones (e.g. a run that
    stopped early) or that lack one of ``variables`` are skipped with a warning.

    Parameters
    ----------
    paths : str or list of str
        Path or list of paths to the NetCDF files, one forecast per file.
    variables : str or list of str, optional
        Variables to load. Defaults to all ``(time, values)`` variables of the
        first file with the most common lead times.
    storage_options : dict of str to Any, optional
        Options passed to ``fsspec.open`` for remote paths (e.g. ``s3://``).

    Returns
    -------
    xr.Dataset
        ``(reference_time, lead_time, grid_index)`` variables with
        latitude/longitude coordinates and the mlwp trait attributes.

    Raises
    ------
    ValueError
        If any of ``variables`` is not a gridded variable of the files.
    """
    paths = [paths] if isinstance(paths, str) else list(paths)
    if isinstance(variables, str):
        variables = [variables]

    valid_times, names = {}, {}
    for path in paths:
        with _open(path, storage_options) as f:
            valid_times[path] = _decode("time", f.variables["time"]).values
            names[path] = set(f.variables)
    lead_times = {path: times - times[0] for path, times in valid_times.items()}
    common, _ = Counter(tuple(lt) for lt in lead_times.values()).most_common(1)[0]
    common = np.array(common)
    reference = next(p for p in paths if np.array_equal(lead_times[p], common))

    with _open(reference, storage_options) as f:
        on_grid = [name for name, var in f.variables.items() if var.dimensions == DIMS]
        if variables is None:
            variables = on_grid
        missing = [name for name in variables if name not in on_grid]
        if missing:
            raise ValueError(f"No {DIMS} variables {missing} in {reference}")
        # Decode zero-size stand-ins to get each variable's dtype and attributes
        templates = {
            name: _decode(name, var, np.empty((0, 0), var.dtype))
            for name in variables
            for var in [f.variables[name]]
        }
        latitude = _decode("latitude", f.variables["latitude"])
        longitude = _decode("longitude", f.variables["longitude"])
        attrs = dict(f.attrs)

    kept = []
    for path in paths:
        missing = [name for name in variables if name not in names[path]]
        if not np.array_equal(lead_times[path], common):
            logger.warning(
                f"Skipping {path}: its lead times differ from the most common "
                f"ones, those of {reference} "
                f"({len(lead_times[path])} vs {len(common)} steps)"
            )
        elif missing:
            logger.warning(f"Skipping {path}: variables {missing} not found")
        else:
            kept.append(path)

    # One chunk per file; each read task gets a block holding its file's path
    path_blocks = da.from_array(np.array(kept), chunks=1)
    chunks = ((1,) * len(kept), (len(common),), (latitude.size,))
    data_vars = {
        name: xr.Variable(
            ("reference_time", "lead_time", "grid_index"),
            path_blocks.map_blocks(
                _read_variable,
                name,
                storage_options=storage_options,
                new_axis=[1, 2],
                chunks=chunks,
                meta=np.empty((0, 0, 0), dtype=template.dtype),
                token=f"anemoi-inference-{name}",
            ),
            attrs=template.attrs,
        )
        for name, template in templates.items()
    }
    latitude.attrs.update({"standard_name": "latitude", "units": "degrees_north"})
    longitude.attrs.update({"standard_name": "longitude", "units": "degrees_east"})
    coords = {
        "reference_time": (
            "reference_time",
            np.array([valid_times[path][0] for path in kept]),
            {"standard_name": "forecast_reference_time"},
        ),
        "lead_time": (
            "lead_time",
            common,
            {"standard_name": "forecast_period", "units": "hours"},
        ),
        "latitude": ("grid_index", latitude.data, latitude.attrs),
        "longitude": ("grid_index", longitude.data, longitude.attrs),
    }

    ds_out = xr.Dataset(data_vars, coords=coords, attrs=attrs)
    ds_out.attrs[TIME_TRAIT_ATTR] = Time.FORECAST
    ds_out.attrs[SPACE_TRAIT_ATTR] = Space.GRID
    ds_out.attrs[UNCERTAINTY_TRAIT_ATTR] = Uncertainty.DETERMINISTIC
    return ds_out


@contextlib.contextmanager
def _open(path: str, storage_options: dict[str, Any] | None) -> Iterator[h5netcdf.File]:
    """Open a NetCDF file with h5netcdf, through fsspec for remote paths.

    Parameters
    ----------
    path : str
        Local path or URL (e.g. ``s3://``) of the file.
    storage_options : dict or None
        Options passed to ``fsspec.open`` for remote paths.

    Yields
    ------
    h5netcdf.File
        The file, open for reading.
    """
    if fsspec.utils.get_protocol(path) == "file":
        with h5netcdf.File(path, "r") as f:
            yield f
    else:
        with fsspec.open(path, mode="rb", **(storage_options or {})) as fobj:
            with h5netcdf.File(fobj, "r") as f:
                yield f


def _decode(
    name: str, var: h5netcdf.Variable, values: np.ndarray | None = None
) -> xr.Variable:
    """Apply CF decoding (fill values, scale/offset, times) the way xarray does.

    Parameters
    ----------
    name : str
        Name of the variable.
    var : h5netcdf.Variable
        Variable whose dimensions and attributes are used.
    values : np.ndarray, optional
        Values to decode in place of the variable's; all of ``var`` is read
        if not given.

    Returns
    -------
    xr.Variable
        The decoded variable.
    """
    raw = xr.Variable(
        var.dimensions, var[...] if values is None else values, dict(var.attrs)
    )
    return xr.conventions.decode_cf_variable(name, raw)


def _read_variable(
    path_block: np.ndarray, name: str, storage_options: dict[str, Any] | None
) -> np.ndarray:
    """Read one variable of one file as a ``(1, time, values)`` block.

    h5netcdf pads variables shorter than the unlimited time dimension, such as
    constant fields written at the first step only, with their fill value.

    Parameters
    ----------
    path_block : np.ndarray
        One-element array holding the path of the file.
    name : str
        Name of the variable to read.
    storage_options : dict or None
        Options passed to ``fsspec.open`` for remote paths.

    Returns
    -------
    np.ndarray
        ``(1, time, values)`` block of the decoded values.
    """
    with _open(str(path_block[0]), storage_options) as f:
        values = _decode(name, f.variables[name]).values
    return values[np.newaxis]
