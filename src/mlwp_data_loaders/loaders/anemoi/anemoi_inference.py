import xarray as xr

TIME_PROFILE = "forecast"
SPACE_PROFILE = "grid"
UNCERTAINTY_PROFILE = "deterministic"


def load_dataset(paths, chunks="auto", engine="h5netcdf", parallel=True, **kwargs):
    """
    Load Anemoi inference datasets from NetCDF/HDF5 files.

    Parameters
    ----------
    paths : str or list of str
        Path or list of paths to the dataset files.
    chunks : str or dict, default: "auto"
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
        The loaded and pre-processed xarray Dataset with lead time coordinates.
    """
    paths = [paths] if isinstance(paths, str) else paths

    # FIXME/TODO: If the multiple files in `paths` are chunked across different times
    # (e.g. file 1 is Jan, file 2 is Feb), extracting `times` from only `paths[0]`
    # means the `lead_times` array will be shorter than the concatenated time dimension.
    # We may need to rethink this, but keeping it for now to match original behavior.
    times = xr.open_dataset(paths[0], engine=engine)["time"].values
    lead_times = times - times[0]

    ds = xr.open_mfdataset(
        paths,
        preprocess=_preprocess,
        chunks=chunks,
        engine=engine,
        parallel=parallel,
        **kwargs,
    )

    ds_out = (
        ds.assign_coords({"lead_time": ("time", lead_times)})
        .rename_dims({"values": "grid_index"})
        .swap_dims({"time": "lead_time"})
    )

    return ds_out


def _preprocess(ds):
    ds_out = (
        ds.set_coords(["longitude", "latitude"])
        .expand_dims("reference_time")
        .assign_coords({"reference_time": ("reference_time", [ds["time"].values[0]])})
        .drop_vars("time")
    )

    return ds_out
