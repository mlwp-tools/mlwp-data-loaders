# mlwp-data-loaders

Loader package for opening source datasets before validating them with
[`mlwp-data-specs`](../mlwp-data-specs/).

## What this package does

`mlwp-data-loaders` is responsible for:

1. importing a loader module or script
2. opening one or more source datasets using loader-defined hooks
3. optionally checking that the chosen trait profiles are compatible with the loader
4. returning an `xarray.Dataset` that can then be validated with `mlwp-data-specs`

The intended split is:

1. `mlwp-data-loaders`: source-specific loading and normalization
2. `mlwp-data-specs`: trait validation

## Python API

The `loader` argument is interpreted as:

- a Python file path if it ends with `.py`
- a Python module path if it contains `.`

Load a dataset through a loader module:

```python
from mlwp_data_loaders import load_dataset
from mlwp_data_specs import validate_dataset

ds = load_dataset(
    [
        "/path/to/anemoi-inference-20260101T00.nc",
        "/path/to/anemoi-inference-20260102T00.nc",
    ],
    loader="mlwp_data_loaders.loaders.anemoi.anemoi_inference",
    time="forecast",
    space="grid",
    uncertainty="deterministic",
)

report = validate_dataset(
    ds,
    time="forecast",
    space="grid",
    uncertainty="deterministic",
)

report.console_print()
```

## CLI

Use the loader-aware CLI:

```bash
uv run mlwp.load_and_validate_dataset \
  /path/to/anemoi-inference-20260101T00.nc \
  /path/to/anemoi-inference-20260102T00.nc \
  --loader mlwp_data_loaders.loaders.anemoi.anemoi_inference \
  --time forecast \
  --space grid \
  --uncertainty deterministic
```

Using a user-provided loader script:

```bash
uv run mlwp.load_and_validate_dataset \
  /path/to/source-a.nc \
  /path/to/source-b.nc \
  --loader ./examples/my_loader.py \
  --time forecast \
  --space grid \
  --uncertainty deterministic
```

## Loader module contract

The loader module may define a subset of the following:

1. Variables defining how each provided path is opened with `xarray.open_dataset`
   - `OPEN_KWARGS`: keyword arguments forwarded to `xarray.open_dataset`, including backend selection such as `{"engine": "zarr"}` or `{"engine": "h5netcdf"}`
2. Functions and variables around preprocessing, concatenation, and postprocessing
   - `preprocess(ds)`: normalize each opened source dataset before combination
   - `CONCAT_DIM`: dimension used when combining multiple inputs; required if more than one path is provided
   - `postprocess(ds)`: finalize the combined dataset before validation
3. Variables defining valid trait profiles
   - `valid_time_profiles`: allowed `time=` profile values for this loader
   - `valid_space_profiles`: allowed `space=` profile values for this loader
   - `valid_uncertainty_profiles`: allowed `uncertainty=` profile values for this loader

Example:

```python
import xarray as xr

OPEN_KWARGS = {}


def preprocess(ds: xr.Dataset) -> xr.Dataset | xr.DataArray:
    return ds


CONCAT_DIM = "valid_time"


def postprocess(ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    return ds
```
