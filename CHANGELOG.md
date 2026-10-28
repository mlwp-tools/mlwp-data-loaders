# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `mlwp_data_loaders.ensemble.apply_member_conventions`: renames an ensemble's member coordinate to `member`, casts its labels to integers, sets the CF `realization` attributes and makes it the leading dimension. Loaders that return ensembles call it last. @mpvginde
- `variables` argument in the IFS forecast loader. @mpvginde

### Changed
- The IFS forecast loader is lazy: each dask chunk holds one lead time of one reference time on the whole grid, with the members grouped up to dask's `array.chunk-size`, so selecting lead times reads only their fields. The old default (`chunks=None`) read all files into memory. **Breaking:** the `chunks` argument is removed. @mpvginde
- The IFS forecast loader returns `(member, reference_time, lead_time, grid_index)`, keeps a scalar `member` for a file holding a single control or perturbed member, and skips files whose lead times, members or variables differ from the most common ones, with a warning. @mpvginde
- `dask` is now a required dependency. @mpvginde

### Fixed
- The IFS forecast loader opens files with cfgrib whatever their extension. @mpvginde

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
