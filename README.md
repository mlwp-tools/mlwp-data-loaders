# mlwp-data-loaders

Loader package for opening source datasets before validating them with
[`mlwp-data-specs`](../mlwp-data-specs/).

## What this package does

`mlwp-data-loaders` is responsible for:

1. Importing a loader module (or Python file).
2. Using the loader to open and normalize source datasets.
3. Extracting the appropriate validation trait profiles (`TIME_PROFILE`, `SPACE_PROFILE`, `UNCERTAINTY_PROFILE`) defined by the loader.
4. Validating the returned `xarray.Dataset` automatically via `mlwp-data-specs`.
5. Returning the `xarray.Dataset` (and optionally the trait dict) for further use or machine learning workloads.

The intended split is:
- **`mlwp-data-loaders`**: Source-specific loading and normalization logic.
- **`mlwp-data-specs`**: General trait validation and compliance checks.

## Python API

The `loader` argument is interpreted as:
- A Python file path if it ends with `.py`.
- A Python module path if it contains `.` (e.g. `mlwp_data_loaders.loaders.anemoi.anemoi_inference`).

You can load a dataset and its trait profiles natively:

```python
from mlwp_data_loaders import load_dataset

ds, dataset_traits = load_dataset(
    [
        "/path/to/anemoi-inference-20260101T00.nc",
        "/path/to/anemoi-inference-20260102T00.nc",
    ],
    loader="mlwp_data_loaders.loaders.anemoi.anemoi_inference",
    return_dataset_traits=True,
)

print(f"Validation time profile used: {dataset_traits['time_profile']}")
print(f"Dataset has {len(ds.data_vars)} variables.")
```

If you don't need the traits dictionary returned, simply omit `return_dataset_traits` (defaults to `False`):

```python
ds = load_dataset(
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

Each loader module must define a function and optionally standard profile variables:

1. `load_dataset(path: str | list[str], **kwargs) -> xr.Dataset`
   - **Required**. Handles opening the path(s), preprocessing, concatenating, and postprocessing, returning a single normalized `xarray.Dataset`.
2. `TIME_PROFILE`: `str`
   - Defines the time trait profile for `mlwp-data-specs` validation (e.g. `"forecast"`).
3. `SPACE_PROFILE`: `str`
   - Defines the space trait profile (e.g. `"grid"`).
4. `UNCERTAINTY_PROFILE`: `str`
   - Defines the uncertainty trait profile (e.g. `"deterministic"`).

### Example Loader (`my_loader.py`)

```python
import xarray as xr

TIME_PROFILE = "observation"
SPACE_PROFILE = "grid"
UNCERTAINTY_PROFILE = "deterministic"

def load_dataset(path: str | list[str], **kwargs) -> xr.Dataset:
    if isinstance(path, list):
        ds = xr.open_mfdataset(path, combine="by_coords", **kwargs)
    else:
        ds = xr.open_dataset(path, **kwargs)

    # Example post-processing
    if "time" in ds.dims:
        ds = ds.rename({"time": "valid_time"})

    return ds
```
