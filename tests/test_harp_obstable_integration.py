"""Integration tests for the built-in ``harp.obstable`` loader, loaded via intake."""

from __future__ import annotations

from pathlib import Path

import intake
import xarray as xr
from mlwp_data_specs.api import (
    SPACE_TRAIT_ATTR,
    TIME_TRAIT_ATTR,
    UNCERTAINTY_TRAIT_ATTR,
)

from mlwp_data_loaders.mxalign_api import validate_dataset_with_mxalign

HERE = Path(__file__).parent
CATALOG = HERE / "catalog" / "test_datasets.yaml"


def test_load_dataset_opens_harp_obstable() -> None:
    """The harp.obstable loader can open and validate the sample SQLite file via intake."""
    cat = intake.open_catalog(str(CATALOG))
    source = cat["harp_obstable"]["observation_table"]

    ds = source.read()
    assert isinstance(ds, xr.Dataset)

    # Note: mxalign validation is temporarily kept here during early development
    # to ensure `mlwp-data-specs` behaves identically. It will eventually be removed.
    report_mxalign = validate_dataset_with_mxalign(
        ds,
        time=ds.attrs.get(TIME_TRAIT_ATTR),
        space=ds.attrs.get(SPACE_TRAIT_ATTR),
        uncertainty=ds.attrs.get(UNCERTAINTY_TRAIT_ATTR),
    )
    if report_mxalign.has_fails():
        report_mxalign.console_print()
    assert not report_mxalign.has_fails()
