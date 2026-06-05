"""Tests for the Intake driver that wraps mlwp-data-loaders loaders."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import intake
import pytest
import xarray as xr

HERE = Path(__file__).parent
CATALOG = HERE / "catalog" / "test_datasets.yaml"


def _make_catalog(tmp_path: Path, loader_path: str, **extra_args) -> intake.Catalog:
    """Write an in-memory catalog YAML to a temp file and open it."""
    import yaml

    catalog = {
        "sources": {
            "test_ds": {
                "driver": "mlwp_loader",
                "args": {
                    "dataset_path": "dummy",
                    "loader": loader_path,
                    **extra_args,
                },
            }
        }
    }
    catalog_file = tmp_path / "catalog.yaml"
    with open(catalog_file, "w") as f:
        yaml.dump(catalog, f)
    return intake.open_catalog(str(catalog_file))


def test_entry_point_is_registered() -> None:
    """The mlwp_loader entry point is discoverable by Intake."""
    eps = importlib.metadata.entry_points(group="intake.drivers")
    matching = [ep for ep in eps if ep.name == "mlwp_loader"]
    assert len(matching) == 1
    cls = matching[0].load()
    assert cls.name == "mlwp_loader"


def test_driver_opens_dataset_from_loader_script(tmp_path) -> None:
    """The driver can load a dataset via a .py loader file."""
    loader_file = tmp_path / "loader.py"
    loader_file.write_text(
        "def load_dataset(path, **kwargs):\n"
        "    import xarray as xr\n"
        "    ds = xr.Dataset({'x': ('t', [1, 2, 3])})\n"
        "    ds.coords['t'] = [0, 1, 2]\n"
        "    ds.attrs['mlwp_time_trait'] = 'observation'\n"
        "    ds.attrs['mlwp_space_trait'] = 'point'\n"
        "    ds.attrs['mlwp_uncertainty_trait'] = 'deterministic'\n"
        "    return ds\n"
    )

    cat = _make_catalog(tmp_path, str(loader_file))
    ds = cat["test_ds"].read()
    assert isinstance(ds, xr.Dataset)
    assert "x" in ds.data_vars
    assert ds["x"].values.tolist() == [1, 2, 3]


def test_test_catalog_structure() -> None:
    """The test datasets catalog has the expected loader-grouped structure."""
    assert CATALOG.exists(), f"Catalog not found at {CATALOG}"

    cat = intake.open_catalog(str(CATALOG))
    loader_groups = list(cat)
    assert "anemoi_datasets" in loader_groups
    assert "anemoi_inference" in loader_groups
    assert "harp_obstable" in loader_groups

    assert list(cat["anemoi_datasets"]) == ["cerra_sample"]
    assert list(cat["anemoi_inference"]) == []
    assert list(cat["harp_obstable"]) == ["observation_table"]


def test_driver_supports_to_dask(tmp_path) -> None:
    """to_dask() returns the same dataset as read()."""
    loader_file = tmp_path / "loader.py"
    loader_file.write_text(
        "def load_dataset(path, **kwargs):\n"
        "    import xarray as xr\n"
        "    ds = xr.Dataset({'x': ('t', [1, 2, 3])})\n"
        "    ds.coords['t'] = [0, 1, 2]\n"
        "    ds.attrs['mlwp_time_trait'] = 'observation'\n"
        "    ds.attrs['mlwp_space_trait'] = 'point'\n"
        "    ds.attrs['mlwp_uncertainty_trait'] = 'deterministic'\n"
        "    return ds\n"
    )

    cat = _make_catalog(tmp_path, str(loader_file))
    source = cat["test_ds"]
    ds_read = source.read()
    ds_dask = source.to_dask()
    assert isinstance(ds_dask, xr.Dataset)
    assert ds_dask.equals(ds_read)


def test_traverse_catalog_and_open_each_with_to_dask(tmp_path) -> None:
    """All entries in a catalog can be opened with to_dask()."""
    import yaml

    # Write a multi-entry catalog
    catalog = {
        "sources": {
            "a": {
                "driver": "mlwp_loader",
                "args": {
                    "dataset_path": "dummy",
                    "loader": str(_make_loader(tmp_path, "a", [1, 2])),
                },
            },
            "b": {
                "driver": "mlwp_loader",
                "args": {
                    "dataset_path": "dummy",
                    "loader": str(_make_loader(tmp_path, "b", [3, 4, 5])),
                },
            },
        }
    }
    catalog_file = tmp_path / "multi.yaml"
    with open(catalog_file, "w") as f:
        yaml.dump(catalog, f)

    cat = intake.open_catalog(str(catalog_file))
    for name in cat:
        ds = cat[name].to_dask()
        assert isinstance(ds, xr.Dataset), f"{name} did not return a Dataset"
        assert name in str(ds.data_vars), f"{name} missing expected data variable"


def _make_loader(tmp_path: Path, name: str, values: list[int]) -> Path:
    """Write a loader script that returns a dataset with a known data variable."""
    loader_file = tmp_path / f"loader_{name}.py"
    loader_file.write_text(
        f"def load_dataset(path, **kwargs):\n"
        f"    import xarray as xr\n"
        f"    ds = xr.Dataset({{'{name}': ('x', {values!r})}})\n"
        f"    ds.coords['x'] = list(range(len(ds['{name}'])))\n"
        f"    ds.attrs['mlwp_time_trait'] = 'observation'\n"
        f"    ds.attrs['mlwp_space_trait'] = 'point'\n"
        f"    ds.attrs['mlwp_uncertainty_trait'] = 'deterministic'\n"
        f"    return ds\n"
    )
    return loader_file


def test_driver_discover_returns_schema(tmp_path) -> None:
    """discover() returns metadata about the dataset without loading fully."""
    loader_file = tmp_path / "loader.py"
    loader_file.write_text(
        "def load_dataset(path, **kwargs):\n"
        "    import xarray as xr\n"
        "    import numpy as np\n"
        "    ds = xr.Dataset({'x': ('t', np.array([1, 2, 3], dtype='f4'))})\n"
        "    ds.coords['t'] = [0, 1, 2]\n"
        "    ds.attrs['mlwp_time_trait'] = 'observation'\n"
        "    ds.attrs['mlwp_space_trait'] = 'point'\n"
        "    ds.attrs['mlwp_uncertainty_trait'] = 'deterministic'\n"
        "    return ds\n"
    )

    cat = _make_catalog(tmp_path, str(loader_file))
    source = cat["test_ds"]
    info = source.discover()
    assert "x" in info["dtype"]
    assert info["npartitions"] == 1
