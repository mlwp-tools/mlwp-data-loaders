# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `load_and_validate_dataset` accepts a Python callable as `loader`, in addition to a `.py` file path or a module path, so a loader function can be passed in directly without writing a module. @mpvginde

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
