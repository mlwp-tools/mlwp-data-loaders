"""Integration tests for the built-in ``harp.obstable`` loader."""

from __future__ import annotations

import pooch
import pytest
from mlwp_data_specs import validate_dataset

from mlwp_data_loaders.api import load_dataset
from mlwp_data_loaders.core import get_dataset_traits_from_loader
from mlwp_data_loaders.mxalign_api import validate_dataset_with_mxalign

HARP_DATA_URL = "https://raw.githubusercontent.com/harphub/harpData/master/inst/OBSTABLE/OBSTABLE_2019.sqlite"
HARP_DATA_HASH = "bdab991c287a41871488456d1a9d697942aa3a612800a88264defa312a9d637b"
LOADER = "mlwp_data_loaders.loaders.harp.obstable"


@pytest.fixture(scope="module")
def obstable_path() -> str:
    """Download and cache the test SQLite dataset."""
    return pooch.retrieve(
        url=HARP_DATA_URL,
        known_hash=HARP_DATA_HASH,
    )


def test_load_dataset_opens_harp_obstable(obstable_path: str) -> None:
    """The harp.obstable loader can open and validate the sample SQLite file."""
    ds = load_dataset(
        obstable_path,
        loader=LOADER,
    )

    traits = get_dataset_traits_from_loader(LOADER)

    report_mxalign = validate_dataset_with_mxalign(
        ds,
        time=traits.get("time_profile"),
        space=traits.get("space_profile"),
        uncertainty=traits.get("uncertainty_profile"),
    )
    if report_mxalign.has_fails():
        report_mxalign.console_print()
    assert not report_mxalign.has_fails()

    report_specs = validate_dataset(
        ds,
        time=traits.get("time_profile"),
        space=traits.get("space_profile"),
        uncertainty=traits.get("uncertainty_profile"),
    )
    if report_specs.has_fails():
        report_specs.console_print()
    assert not report_specs.has_fails()
