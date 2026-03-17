"""Helpers for validating datasets with local mxalign property checks."""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import xarray as xr
from mlwp_data_specs.specs.reporting import ValidationReport


@lru_cache(maxsize=1)
def _load_mxalign_validation_symbols():
    """Load mxalign property validation modules from the local checkout."""
    base_dir = Path(__file__).resolve().parents[2] / "mxalign" / "src" / "mxalign"
    properties_dir = base_dir / "properties"
    if not properties_dir.exists():
        raise ImportError(f"mxalign checkout not found at {base_dir}")

    package_modules = {
        "mxalign": base_dir,
        "mxalign.properties": properties_dir,
    }
    for module_name, module_path in package_modules.items():
        if module_name in sys.modules:
            continue
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        module = importlib.util.module_from_spec(spec)
        module.__path__ = [str(module_path)]  # type: ignore[attr-defined]
        sys.modules[module_name] = module

    module_paths = {
        "mxalign.properties.properties": properties_dir / "properties.py",
        "mxalign.properties.specs": properties_dir / "specs.py",
        "mxalign.properties.validation": properties_dir / "validation.py",
    }
    for module_name, module_path in module_paths.items():
        if module_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load {module_name} from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    properties_module = sys.modules["mxalign.properties.properties"]
    validation_module = sys.modules["mxalign.properties.validation"]
    return {
        "Properties": properties_module.Properties,
        "Space": properties_module.Space,
        "Time": properties_module.Time,
        "Uncertainty": properties_module.Uncertainty,
        "validate_dataset": validation_module.validate_dataset,
    }


def validate_dataset_with_mxalign(
    ds: xr.Dataset | xr.DataArray,
    *,
    time: str | None = None,
    space: str | None = None,
    uncertainty: str | None = None,
) -> ValidationReport:
    """Validate a dataset with mxalign property checks when selectors are known."""
    report = ValidationReport()
    if time is None or space is None:
        return report

    mxalign = _load_mxalign_validation_symbols()
    properties = mxalign["Properties"](
        time=mxalign["Time"](time),
        space=mxalign["Space"](space),
        uncertainty=mxalign["Uncertainty"](uncertainty or "deterministic"),
    )

    try:
        mxalign["validate_dataset"](ds, properties)
    except ValueError as exc:
        report.add(
            "MXAlign Properties",
            "mxalign.properties.validation.validate_dataset",
            "FAIL",
            str(exc),
        )
    else:
        report.add(
            "MXAlign Properties",
            "mxalign.properties.validation.validate_dataset",
            "PASS",
        )
    return report
