import numpy as np
import xarray as xr
from loguru import logger

TIME_PROFILE = "observation"
SPACE_PROFILE = "grid"
UNCERTAINTY_PROFILE = "deterministic"

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

COORDS = dict(longitude="longitudes", latitude="latitudes", valid_time="dates")


def load_dataset(
    path,
    chunks="auto",
    consolidated=False,
    variables=None,
    storage_options=None,
    **kwargs,
):
    """
    Load Anemoi datasets from Zarr files.

    Parameters
    ----------
    path : str
        Path to the Zarr dataset.
    chunks : str or dict, default: "auto"
        Chunk size or strategy for dask arrays.
    consolidated : bool, default: False
        Whether to use consolidated metadata when opening the Zarr store.
    variables : str or list of str, optional
        List of variables to select from the dataset. If None, all variables are kept.

    Returns
    -------
    xr.Dataset
        The loaded and post-processed xarray Dataset.
    """
    variables = [variables] if isinstance(variables, str) else variables

    ds = xr.open_zarr(
        path,
        consolidated=consolidated,
        chunks=chunks,
        storage_options=storage_options,
        **kwargs,
    )  # type: ignore
    ds_postproc = _postprocess(ds)

    if variables:
        ds_selected = ds_postproc.sel(variable=variables)
    else:
        ds_selected = ds_postproc
        if len(ds_selected["variable"]) > 10:
            logger.info(
                f"Transforming anemoi-datasets xr.DataArray with {len(ds_postproc['variable'])} variables "
                f"to xr.Dataset, this might take some time. Consider selecting the relevant variables during loading"
            )

    return ds_selected.to_dataset(dim="variable")


def _postprocess(dataset: xr.Dataset) -> xr.Dataset:
    """Post-process the dataset to add coordinates and drop unused variables.

    Args:
        dataset (xr.Dataset): The input dataset to be processed.

    Returns:
        xr.Dataset: The processed dataset with assigned coordinates and
            attributes.
    """

    # Add coordinates
    coords = {
        key: dataset[value].astype("datetime64[ns]").load()
        if key == "valid_time"
        else dataset[value].load()
        for key, value in COORDS.items()
    }
    for key in ("latitude", "longitude"):
        coords[key] = coords[key].astype(np.float32)

    coords["variable"] = dataset.attrs["variables"]
    coords["valid_time"] = coords["valid_time"].astype("datetime64[ns]")
    ds_coords = dataset.assign_coords(coords)

    # Drop unused variables and remove ensemble dimension
    drop_vars = [var for var in DROP_VARS if var in coords["variable"]]

    ds_pruned = (
        ds_coords["data"]
        .isel(ensemble=0)
        .drop_sel(variable=drop_vars)
        .swap_dims({"time": "valid_time"})
        .rename({"cell": "grid_index"})
    )
    return ds_pruned  # type: ignore
