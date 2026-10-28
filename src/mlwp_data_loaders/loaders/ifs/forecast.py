"""
Loader for IFS forecasts as retrieved from MARS as GRIB files.

Each file holds the fields of one or more forecasts (reference times) for one
or more lead times (``step``) and, for an ensemble, members (``number``). If a
file holds more than one type of vertical level, select one with
``backend_kwargs``, e.g. ``backend_kwargs={"filter_by_keys": {"typeOfLevel":
"surface"}}``. The same goes for an ensemble file that holds both the control
forecast (``dataType`` ``cf``) and the perturbed members (``pf``), which cfgrib
cannot open as one dataset.

The returned dataset is lazy and dask-backed. A GRIB message is one whole
field (one lead time of one member of one variable), and a dask chunk is read
whole even when only part of it is selected later. Every chunk is therefore a
set of whole fields that are used together:

- ``grid_index`` is never split: interpolating to points needs whole fields,
  and splitting them would not save any reading.
- Each reference time and each lead time is its own chunk, so selecting lead
  times (e.g. those with observations) reads no other lead times.
- The members of one lead time share a chunk, as far as they fit in dask's
  ``array.chunk-size`` (128 MiB by default; e.g. 5 members per chunk for ENS
  at O1280), so ensemble statistics computed after interpolating to points
  need little or no rechunking. Set it with
  ``dask.config.set({"array.chunk-size": ...})``.
"""

from collections import Counter
from typing import Any

import dask
import numpy as np
import xarray as xr
from dask.utils import parse_bytes
from loguru import logger
from mlwp_data_specs.api import (
    SPACE_TRAIT_ATTR,
    TIME_TRAIT_ATTR,
    UNCERTAINTY_TRAIT_ATTR,
    Space,
    Time,
    Uncertainty,
)

from mlwp_data_loaders.ensemble import apply_member_conventions

try:
    import cfgrib  # noqa: F401
except ImportError as e:
    raise ImportError(
        "Please install the cfgrib package to load IFS-Forecasts. This can be done by installing the `ifs` optional extra dependencies of `mlwp-data-loaders` with e.g. `pip install mlwp-data-loaders[ifs]`"
    ) from e

# GRIB data types of a single ensemble member: control and perturbed forecast
MEMBER_DATA_TYPES = {"cf", "pf"}


def load_dataset(
    paths: str | list[str],
    variables: str | list[str] | None = None,
    **kwargs: Any,
) -> xr.Dataset:
    """
    Load IFS forecast GRIB files as one lazy, dask-backed dataset.

    Files whose lead times, members or variables differ from the most common
    ones (e.g. a run that stopped early) are skipped with a warning.

    Parameters
    ----------
    paths : str or list of str
        Path or list of paths to the GRIB files.
    variables : str or list of str, optional
        Variables to load, by their cfgrib names (e.g. ``t2m``, ``u10``).
        Defaults to all variables.
    **kwargs
        Additional keyword arguments passed to ``xr.open_dataset`` with the
        cfgrib engine, e.g. ``backend_kwargs={"filter_by_keys": {...}}``.

    Returns
    -------
    xr.Dataset
        ``([member,] reference_time, lead_time, grid_index)`` variables with
        latitude/longitude coordinates and the mlwp trait attributes.
    """
    if "chunks" in kwargs:
        raise TypeError(
            "The IFS forecast loader no longer takes `chunks`: each dask chunk "
            "holds one lead time of one reference time, with the members grouped "
            "up to dask's `array.chunk-size`."
        )
    paths = [paths] if isinstance(paths, str) else list(paths)
    if isinstance(variables, str):
        variables = [variables]

    forecasts = {path: _open_forecast(path, variables, **kwargs) for path in paths}
    layouts = {path: _layout(ds) for path, ds in forecasts.items()}
    common, _ = Counter(layouts.values()).most_common(1)[0]
    reference = next(path for path in paths if layouts[path] == common)

    missing = [name for name in variables or [] if name not in common[-1]]
    if missing:
        raise ValueError(f"Variables {missing} not found in {reference}")

    kept = []
    for path in paths:
        if layouts[path] == common:
            kept.append(path)
        else:
            logger.warning(
                f"Skipping {path} ({_describe(layouts[path])}): its lead times, "
                f"members, grid or variables differ from the most common ones, "
                f"those of {reference} ({_describe(common)})"
            )

    grid = forecasts[reference]
    chunks = _field_chunks(grid)
    ds = xr.concat(
        [_chunk(forecasts[path], chunks) for path in kept],
        dim="reference_time",
        data_vars="all",
        coords="minimal",
        compat="override",
        join="exact",
        combine_attrs="override",
    )

    longitude = (grid["longitude"].values + 180.0) % 360.0 - 180.0
    ds = ds.assign_coords(
        latitude=(
            "grid_index",
            grid["latitude"].values,
            {"standard_name": "latitude", "units": "degrees_north"},
        ),
        longitude=(
            "grid_index",
            longitude,
            {"standard_name": "longitude", "units": "degrees_east"},
        ),
    )
    ds.coords["reference_time"].attrs["standard_name"] = "forecast_reference_time"
    ds.coords["lead_time"].attrs.update(
        {"standard_name": "forecast_period", "units": "hours"}
    )

    ds = apply_member_conventions(ds.transpose("reference_time", "lead_time", ...))

    ds.attrs[TIME_TRAIT_ATTR] = Time.FORECAST
    ds.attrs[SPACE_TRAIT_ATTR] = Space.GRID
    ds.attrs[UNCERTAINTY_TRAIT_ATTR] = (
        Uncertainty.ENSEMBLE if "member" in ds.dims else Uncertainty.DETERMINISTIC
    )
    return ds


def _open_forecast(path: str, variables: list[str] | None, **kwargs: Any) -> xr.Dataset:
    """Open one GRIB file lazily, with the dims renamed.

    cfgrib drops dimensions of length one, so ``reference_time`` and
    ``lead_time`` can be scalar here; ``_chunk`` adds them back. A scalar
    ``number`` is kept for a single ensemble member, which becomes a scalar
    ``member`` coordinate, and dropped otherwise (e.g. for HRES).
    """
    ds = xr.open_dataset(path, engine="cfgrib", **kwargs)
    # valid_time is reference_time + lead_time; mxalign derives it when needed
    ds = ds.drop_vars(["valid_time", "surface"], errors="ignore")
    if variables is not None:
        ds = ds[[name for name in variables if name in ds.data_vars]]

    if "number" in ds.coords and "number" not in ds.dims:
        data_types = {var.attrs.get("GRIB_dataType") for var in ds.data_vars.values()}
        if not data_types <= MEMBER_DATA_TYPES:
            ds = ds.drop_vars("number")

    return ds.rename(
        {"time": "reference_time", "step": "lead_time", "values": "grid_index"}
    )


def _layout(ds: xr.Dataset) -> tuple:
    """Return what must match between files to concatenate them."""
    members = tuple(np.atleast_1d(ds["number"].values)) if "number" in ds.coords else ()
    return (
        tuple(np.atleast_1d(ds["lead_time"].values)),
        members,
        ds.sizes["grid_index"],
        tuple(sorted(ds.data_vars)),
    )


def _describe(layout: tuple) -> str:
    """Describe a file layout from ``_layout`` for a log message."""
    lead_times, members, n_grid, names = layout
    return (
        f"{len(lead_times)} lead times, members {list(members)}, "
        f"{n_grid} grid points, variables {list(names)}"
    )


def _field_chunks(ds: xr.Dataset) -> dict[str, int]:
    """Return chunk sizes of one lead time, whole fields and as many members as fit.

    Members are grouped up to dask's ``array.chunk-size``.
    """
    chunks = {"reference_time": 1, "lead_time": 1, "grid_index": -1}
    if "number" in ds.dims:
        itemsize = max(var.dtype.itemsize for var in ds.data_vars.values())
        field_bytes = ds.sizes["grid_index"] * itemsize
        limit = parse_bytes(dask.config.get("array.chunk-size"))
        chunks["number"] = min(ds.sizes["number"], max(1, limit // field_bytes))
    return chunks


def _chunk(ds: xr.Dataset, chunks: dict[str, int]) -> xr.Dataset:
    """Chunk the variables of one file and add its missing length-one dims.

    The dims are added after chunking because expanding a variable that is
    not yet a dask array reads all of it.
    """
    ds = ds.drop_vars(["latitude", "longitude"])
    ds = ds.chunk({dim: size for dim, size in chunks.items() if dim in ds.dims})
    return ds.expand_dims(
        [dim for dim in ("reference_time", "lead_time") if dim not in ds.dims]
    )
