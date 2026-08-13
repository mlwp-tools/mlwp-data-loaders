"""Integration tests for the built-in ``harp.obstable`` loader."""

from __future__ import annotations

import pooch
import pytest

from mlwp_data_loaders.api import load_and_validate_dataset

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
    ds, report_specs = load_and_validate_dataset(  # type: ignore
        obstable_path,
        loader=LOADER,
        return_validation_report=True,
    )

    if report_specs.has_fails():
        report_specs.console_print()
    assert not report_specs.has_fails()
