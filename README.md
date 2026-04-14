# mlwp-data-loaders

Loader package for opening source datasets before validating them with
[`mlwp-data-specs`](../mlwp-data-specs/).

## What this package does

`mlwp-data-loaders` is responsible for:

1. Importing a loader module (or Python file).
2. Using the loader to open and normalize source files, while [setting global attributes on the resulting dataset](#loader-module-contract) to indicate the [dataset's traits](https://github.com/mlwp-tools/mlwp-data-specs).
3. Validating the returned dataset automatically via `mlwp-data-specs`.
4. Returning the `xarray.Dataset` (and optionally a validation report) for further use or machine learning workloads.

The intended split is:
- **`mlwp-data-loaders`**: Source-specific loading and normalization logic.
- **`mlwp-data-specs`**: General trait validation and compliance checks.

## Python API

The `loader` argument is interpreted as:
- A Python file path if it ends with `.py`.
- A Python module path if it contains `.` (e.g. `mlwp_data_loaders.loaders.anemoi.anemoi_inference`).

You can load a dataset and get its validation report natively:

```python
from mlwp_data_loaders import load_and_validate_dataset
from mlwp_data_specs import validate_dataset

# 1. Load the dataset and extract the validation report
ds, validation_report = load_and_validate_dataset(
    [
        "/path/to/anemoi-inference-20260101T00.nc",
        "/path/to/anemoi-inference-20260102T00.nc",
    ],
    loader="mlwp_data_loaders.loaders.anemoi.anemoi_inference",
    return_validation_report=True,
)

# 2. Print the validation results to the console
validation_report.console_print()
```

If you don't need the report returned, simply omit `return_validation_report` (defaults to `False`). The function will raise a `ValueError` if the dataset does not pass the validation.

```python
ds = load_and_validate_dataset(
    "s3://my-bucket/dataset.zarr",
    loader="mlwp_data_loaders.loaders.anemoi.anemoi_datasets",
    storage_options={"anon": True},
)
```

## CLI

Use the loader-aware CLI to load and validate data from the command line:

```bash
uv run mlwp.load_and_validate_dataset \
  /path/to/anemoi-inference-20260101T00.nc \
  /path/to/anemoi-inference-20260102T00.nc \
  --loader mlwp_data_loaders.loaders.anemoi.anemoi_inference
```

Using a user-provided custom loader script:

```bash
uv run mlwp.load_and_validate_dataset \
  /path/to/source-a.nc \
  /path/to/source-b.nc \
  --loader ./examples/my_loader.py
```

## Loader Module Contract

Each loader module must define a function and assign the correct trait profile attributes to the dataset:

1. `load_dataset(path: str | list[str], **kwargs) -> xr.Dataset`
   - **Required**. Handles opening the path(s), preprocessing, concatenating, and postprocessing, returning a single normalized `xarray.Dataset`.
2. Attributes attached to the dataset
   - Must set `mlwp_time_trait` (e.g. `"forecast"`).
   - Must set `mlwp_space_trait` (e.g. `"grid"`).
   - Must set `mlwp_uncertainty_trait` (e.g. `"deterministic"`).

### Example Loader (`my_loader.py`)

```python
import xarray as xr
from mlwp_data_specs.api import SPACE_TRAIT_ATTR, TIME_TRAIT_ATTR, UNCERTAINTY_TRAIT_ATTR

def load_dataset(path: str | list[str], **kwargs) -> xr.Dataset:
    if isinstance(path, list):
        ds = xr.open_mfdataset(path, combine="by_coords", **kwargs)
    else:
        ds = xr.open_dataset(path, **kwargs)

    # Example post-processing
    if "time" in ds.dims:
        ds = ds.rename({"time": "valid_time"})

    # Assign required traits for validation
    ds.attrs[TIME_TRAIT_ATTR] = "observation"
    ds.attrs[SPACE_TRAIT_ATTR] = "grid"
    ds.attrs[UNCERTAINTY_TRAIT_ATTR] = "deterministic"

    return ds
```
