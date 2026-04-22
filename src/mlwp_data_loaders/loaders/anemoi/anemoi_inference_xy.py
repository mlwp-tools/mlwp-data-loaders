from typing import Any

import xarray as xr
from mlwp_data_specs.api import (
    SPACE_TRAIT_ATTR,
    TIME_TRAIT_ATTR,
    UNCERTAINTY_TRAIT_ATTR,
)

_VARS_TO_DROP = ["forecast_reference_time", "projection"]


def load_dataset(
    paths: str | list[str],
    chunks: str | dict | None = "auto",
    engine: str = "h5netcdf",
    parallel: bool = True,
    **kwargs: Any,
) -> xr.Dataset:
    """
    Load Anemoi inference datasets (x/y structured grid format) from NetCDF/HDF5 files.

    This loader handles the Anemoi inference output format where the spatial domain
    is represented as a structured 2-D projected grid with ``y`` and ``x`` dimensions,
    and ``latitude``/``longitude`` as 2-D non-dimension coordinates. The forecast
    reference time is stored as a scalar data variable named
    ``forecast_reference_time``.

    Parameters
    ----------
    paths : str or list of str
        Path or list of paths to the dataset files.
    chunks : str or dict or None, default: "auto"
        Chunk size or strategy for dask arrays.
    engine : str, default: "h5netcdf"
        Engine to use for reading the files.
    parallel : bool, default: True
        Whether to open files in parallel using dask.
    **kwargs
        Additional keyword arguments passed to `xr.open_mfdataset`.

    Returns
    -------
    xr.Dataset
        The loaded and pre-processed xarray Dataset.  The returned dataset has:

        - ``reference_time`` as a dimension coordinate.
        - ``lead_time`` as the time dimension (replacing the original ``time``
          dimension), computed as ``time - forecast_reference_time``.
        - ``latitude`` and ``longitude`` as 2-D ``(y, x)`` non-dimension
          coordinates.
    """
    paths = [paths] if isinstance(paths, str) else paths

    ds = xr.open_mfdataset(
        paths,
        preprocess=_preprocess,
        chunks=chunks,
        engine=engine,
        parallel=parallel,
        **kwargs,
    )

    ds.attrs[TIME_TRAIT_ATTR] = "forecast"
    ds.attrs[SPACE_TRAIT_ATTR] = "grid"
    ds.attrs[UNCERTAINTY_TRAIT_ATTR] = "deterministic"

    return ds


def _preprocess(ds: xr.Dataset) -> xr.Dataset:
    """Preprocess individual datasets before concatenation.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset to preprocess.

    Returns
    -------
    xr.Dataset
        The preprocessed dataset with ``reference_time`` as a new dimension,
        ``lead_time`` replacing the ``time`` dimension, and ``latitude``/
        ``longitude`` promoted to coordinates.
    """
    reference_time = ds["forecast_reference_time"].values
    lead_times = ds["time"].values - reference_time

    # Promote latitude/longitude to coordinates if they are data variables
    coord_vars = [v for v in ["latitude", "longitude"] if v in ds.data_vars]
    if coord_vars:
        ds = ds.set_coords(coord_vars)

    drop_vars = [v for v in _VARS_TO_DROP if v in ds]

    ds_out = (
        ds.drop_vars(drop_vars)
        .assign_coords({"lead_time": ("time", lead_times)})
        .expand_dims("reference_time")
        .assign_coords({"reference_time": ("reference_time", [reference_time])})
        .swap_dims({"time": "lead_time"})
        .drop_vars("time")
    )

    return ds_out
