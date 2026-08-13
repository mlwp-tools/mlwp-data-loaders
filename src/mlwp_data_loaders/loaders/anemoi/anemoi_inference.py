from typing import Any

import xarray as xr
from mlwp_data_specs.api import (
    SPACE_TRAIT_ATTR,
    TIME_TRAIT_ATTR,
    UNCERTAINTY_TRAIT_ATTR,
)


def load_dataset(
    paths: str | list[str],
    chunks: str | dict | None = "auto",
    engine: str = "h5netcdf",
    parallel: bool = True,
    storage_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> xr.Dataset:
    """
    Load Anemoi inference datasets from NetCDF/HDF5 files.

    Parameters
    ----------
    paths : str or list of str
        Path or list of paths to the dataset files.
    chunks : str or dict or None, default: "auto"
        Chunk size or strategy for dask arrays.
    engine : str, default: "h5netcdf"
        Engine to use for reading the files.
    parallel : bool, default: True
        Kept for API compatibility.
    storage_options : dict of str to Any, optional
        Storage options passed to xarray when opening remote files.
    **kwargs
        Additional keyword arguments passed to `xr.open_dataset`.

    Returns
    -------
    xr.Dataset
        The loaded and pre-processed xarray Dataset with lead time coordinates.
    """
    paths = [paths] if isinstance(paths, str) else paths

    # FIXME/TODO: If the multiple files in `paths` are chunked across different times
    # (e.g. file 1 is Jan, file 2 is Feb), extracting `times` from only `paths[0]`
    # means the `lead_times` array will be shorter than the concatenated time dimension.
    # We may need to rethink this, but keeping it for now to match original behavior.
    times = xr.open_dataset(paths[0], engine=engine, storage_options=storage_options)[
        "time"
    ].values
    lead_times = times - times[0]

    datasets = [
        _preprocess(
            xr.open_dataset(
                path,
                chunks=chunks,
                engine=engine,
                storage_options=storage_options,
                **kwargs,
            )
        )
        for path in paths
    ]

    ds = (
        datasets[0] if len(datasets) == 1 else xr.concat(datasets, dim="reference_time")
    )

    ds_out = (
        ds.assign_coords({"lead_time": ("time", lead_times)})
        .rename_dims({"values": "grid_index"})
        .swap_dims({"time": "lead_time"})
    )

    ds_out.coords["reference_time"].attrs["standard_name"] = "forecast_reference_time"
    ds_out.coords["lead_time"].attrs.update(
        {"standard_name": "forecast_period", "units": "hours"}
    )
    ds_out.coords["longitude"].attrs.update(
        {"standard_name": "longitude", "units": "degrees_east"}
    )
    ds_out.coords["latitude"].attrs.update(
        {"standard_name": "latitude", "units": "degrees_north"}
    )

    ds_out.attrs[TIME_TRAIT_ATTR] = "forecast"
    ds_out.attrs[SPACE_TRAIT_ATTR] = "grid"
    ds_out.attrs[UNCERTAINTY_TRAIT_ATTR] = "deterministic"

    return ds_out


def _preprocess(ds: xr.Dataset) -> xr.Dataset:
    """Preprocess individual datasets before concatenation.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset to preprocess.

    Returns
    -------
    xr.Dataset
        The preprocessed dataset with reference time expanded.
    """
    ds_out = (
        ds.set_coords(["longitude", "latitude"])
        .expand_dims("reference_time")
        .assign_coords({"reference_time": ("reference_time", [ds["time"].values[0]])})
        .drop_vars("time")
    )

    return ds_out
