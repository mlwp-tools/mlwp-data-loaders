# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `variables` argument in the `anemoi-inference` loader to load a subset of the variables. @mpvginde

### Changed
- The `anemoi-inference` loader reads only `time` from each file when loading, through h5netcdf directly, instead of opening every file with `xr.open_dataset` (about 3 s per file with 70 variables). Its dask graph no longer depends on the number of files: one layer per variable, with one chunk per file. **Breaking:** the `chunks`, `engine` and `parallel` arguments and the keyword arguments forwarded to `xr.open_dataset` are removed. @mpvginde
- The `anemoi-inference` loader skips files whose lead times differ from the most common ones (e.g. a run that stopped early) or that lack a requested variable, with a warning, instead of failing. @mpvginde
- `dask` is now a required dependency. @mpvginde

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
