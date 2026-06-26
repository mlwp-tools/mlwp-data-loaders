"""
Loader for IFS-forecasts as provided by the ECMWF and typically retrieved
from MARS as grib-file(s).
These grib files typically have dimensions (step, values and optionally number).
If the grib file contains more than one type of vertical level please provide the ``backend_kwarg``.
E.g. ```backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface",}}```
"""


import importlib.util
import warnings
from typing import Any

import xarray as xr
from mlwp_data_specs.api import (
    SPACE_TRAIT_ATTR,
    TIME_TRAIT_ATTR,
    UNCERTAINTY_TRAIT_ATTR,
)

try:
    import cfgrib  # noqa: F401
except ImportError as e:
    raise ImportError("Please install the cfgrib package to load IFS-Forecasts") from e


def load_dataset(
    paths: str | list[str],
    chunks: dict | None = None,  # type: ignore[assignment]
    **kwargs: Any,
) -> xr.Dataset:
    """
    Load IFS forecast datasets from GRIB files.

    Parameters
    ----------
    paths : str or list of str
        Path or list of paths to the GRIB files.
    chunks : dict or None, optional
        Chunk sizes for dask arrays. Pass ``None`` to load eagerly without chunking.
        Defaults to ``{"time": 1, "step": -1, "values": -1}``.
    **kwargs
        Additional keyword arguments passed to `xr.open_mfdataset` (or `xr.open_dataset`
        when ``chunks=None``).

    Returns
    -------
    xr.Dataset
        The loaded and pre-processed xarray Dataset with renamed dimensions and coordinates.
    """
    paths = [paths] if isinstance(paths, str) else paths

    if chunks is not None and not _dask_is_available():
        warnings.warn(
            "dask is not installed but `chunks` was provided. Chunked loading via "
            "`xr.open_mfdataset` requires dask. Falling back to eager loading "
            "(equivalent to `chunks=None`). Install dask to enable chunked loading.",
            UserWarning,
            stacklevel=2,
        )
        chunks = None

    if chunks is None:
        # open_mfdataset requires dask; open individually and concat when chunks=None
        ds = xr.concat(
            [
                _drop_valid_time_var(xr.open_dataset(p, engine="cfgrib", **kwargs))
                for p in paths
            ],
            dim="time",
            coords="minimal",
        )
    else:
        ds = xr.open_mfdataset(
            paths,
            preprocess=_drop_valid_time_var,
            combine="nested",
            concat_dim="time",
            chunks=chunks,
            **kwargs,
        )

    ds.coords["longitude"] = (ds.coords["longitude"] + 180.0) % 360.0 - 180.0

    rename_dims = {
        "time": "reference_time",
        "step": "lead_time",
        "values": "grid_index",
    }
    rename_vars = {
        "time": "reference_time",
        "step": "lead_time",
    }

    if "number" in ds.dims and "number" in ds.coords:
        rename_dims["number"] = "member"
        rename_vars["number"] = "member"
    else:
        ds = ds.drop_vars("number", errors="ignore")

    ds = ds.rename_dims({k: v for k, v in rename_dims.items() if k in ds.dims})
    ds = ds.rename_vars({k: v for k, v in rename_vars.items() if k in ds.variables})

    if "surface" in ds.variables:
        ds = ds.drop_vars("surface")

    ds.coords["lead_time"].attrs.update(
        {"standard_name": "forecast_period", "units": "hours"}
    )

    if "member" in ds.dims:
        uncertainty = "ensemble"
    elif "quantile" in ds.dims:
        uncertainty = "quantile"
    else:
        uncertainty = "deterministic"

    ds = ds.transpose("reference_time", "lead_time", ...)

    ds.attrs[TIME_TRAIT_ATTR] = "forecast"
    ds.attrs[SPACE_TRAIT_ATTR] = "grid"
    ds.attrs[UNCERTAINTY_TRAIT_ATTR] = uncertainty

    return ds


def _dask_is_available() -> bool:
    """Return whether dask is installed without importing it."""
    return importlib.util.find_spec("dask") is not None


def _drop_valid_time_var(ds: xr.Dataset) -> xr.Dataset:
    """Drop valid_time before concatenation.

    valid_time is reference_time + lead_time and differs across files with different
    reference times, making it incompatible as a shared coordinate during concat.
    """
    return ds.drop_vars("valid_time", errors="ignore")
