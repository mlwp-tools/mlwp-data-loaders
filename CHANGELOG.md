# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- numpy-style docstrings are enforced in `src/` with numpydoc's `numpydoc-validation` pre-commit hook. [\#32](https://github.com/mlwp-tools/mlwp-data-loaders/pull/32), @leifdenby
- `mlwp_data_loaders.ensemble.apply_member_conventions`: renames an ensemble's member coordinate to `member`, casts its labels to integers, sets the CF `realization` attributes and makes it the leading dimension. Loaders that return ensembles call it last. @mpvginde
- `variables` argument in the IFS forecast loader. @mpvginde
- `variables` argument in the `anemoi-inference` loader to load a subset of the variables. @mpvginde
- `valid_times` argument (label slice) in the `anemoi-datasets` loader to load part of the record. @mpvginde

### Fixed
- The IFS forecast loader opens files with cfgrib whatever their extension. @mpvginde

### Changed
- The IFS forecast loader is lazy: each dask chunk holds one lead time of one reference time on the whole grid, with the members grouped up to dask's `array.chunk-size`, so selecting lead times reads only their fields. The old default (`chunks=None`) read all files into memory. **Breaking:** the `chunks` argument is removed. @mpvginde
- The IFS forecast loader returns `(member, reference_time, lead_time, grid_index)`, keeps a scalar `member` for a file holding a single control or perturbed member, and skips files whose lead times, members or variables differ from the most common ones, with a warning. @mpvginde
- `dask` is now a required dependency. @mpvginde
- The `anemoi-inference` loader reads only `time` from each file when loading, through h5netcdf directly, instead of opening every file with `xr.open_dataset` (about 3 s per file with 70 variables). Its dask graph no longer depends on the number of files: one layer per variable, with one chunk per file. **Breaking:** the `chunks`, `engine` and `parallel` arguments and the keyword arguments forwarded to `xr.open_dataset` are removed. @mpvginde
- The `anemoi-inference` loader skips files whose lead times differ from the most common ones (e.g. a run that stopped early) or that lack a requested variable, with a warning, instead of failing. @mpvginde
- `dask` is now a required dependency. @mpvginde
- The `anemoi-datasets` loader builds a dask graph whose size no longer depends on the length of the record: all variables are split lazily from one shared read, so each on-disk chunk is read once per compute, using zarr's fast basic-indexing path. **Breaking:** the `chunks` argument is removed; dask chunks follow the store's on-disk time chunks. @michielv
- `dask` is now a required dependency. @mpvginde
- `load_and_validate_dataset` accepts a Python callable as `loader`, in addition to a `.py` file path or a module path, so a loader function can be passed in directly without writing a module. @mpvginde
- The HARP obstable loader is lazy: loading reads only the valid times and the station coordinates, instead of the whole table into memory. Each dask chunk holds a range of valid times of all stations, as many as fit in dask's `array.chunk-size` with all requested variables, read with one indexed query; the variables are split from that shared read, so the graph has `2 + n_variables` layers whatever the length of the record. `variables` can also be a single name, and `valid_time` is `datetime64[ns]`. @mpvginde
- The HARP obstable loader takes each station's latitude, longitude and altitude from its latest row. **Breaking:** they change for stations whose coordinates vary within the table, whose latitude and longitude were the minimum over all rows and whose altitude came from an arbitrary row. @mpvginde

### Fixed
- The HARP obstable loader opens the file read-only and closes it again: a missing path raises `FileNotFoundError` instead of creating an empty database. Unknown `variables` raise a `ValueError`, and column names are quoted in SQL, so a name such as `10si` is no longer read as the number 10. @mpvginde

## [0.1.0](https://github.com/mlwp-tools/mlwp-data-loaders/releases/tag/v0.1.0)

First release of `mlwp-data-loaders`, a package for loading raw MLWP source
datasets into [`mlwp-data-specs`](https://github.com/mlwp-tools/mlwp-data-specs)-conforming
`xarray.Dataset` objects, with automatic trait validation. It bundles
loaders for Anemoi datasets output, Anemoi inference output, HARP SQLite
observation tables, and IFS forecast GRIB output (via the optional `ifs`
extra), exposed through both a Python API
(`mlwp_data_loaders.load_and_validate_dataset(...)`, with an optional
validation report) and a CLI entry point (`mlwp.load_and_validate_dataset`).
Loaders can be bundled (by dotted module path) or user-provided (by `.py`
file path); a loader module implements `load_dataset(path, **kwargs) ->
xr.Dataset` and sets the required trait attributes so the result is
validated automatically.
