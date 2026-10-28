"""Tests for the ensemble member conventions helper."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from mlwp_data_loaders.ensemble import MEMBER_ATTRS, apply_member_conventions


def _ensemble(labels) -> xr.Dataset:
    """Return a ``(reference_time, number, grid_index)`` dataset with member labels ``labels``."""
    data = np.zeros((2, len(labels), 4))
    return xr.Dataset(
        {"t2m": (("reference_time", "number", "grid_index"), data)},
        coords={"number": ("number", labels, {"long_name": "source name"})},
    )


def test_renames_and_moves_member_first() -> None:
    """``number`` becomes ``member``, the leading dim, with the CF attributes."""
    ds = apply_member_conventions(_ensemble([5, 7, 9]))

    assert ds["t2m"].dims == ("member", "reference_time", "grid_index")
    assert "number" not in ds.variables
    assert ds["member"].values.tolist() == [5, 7, 9]
    assert ds["member"].attrs == MEMBER_ATTRS


def test_casts_string_labels_to_int() -> None:
    """Numeric string labels become integers, without renumbering."""
    ds = apply_member_conventions(_ensemble(np.array(["0", "3", "4"])))

    assert ds["member"].dtype == np.int64
    assert ds["member"].values.tolist() == [0, 3, 4]


def test_rejects_non_numeric_labels() -> None:
    """Labels that are not numbers cannot follow the integer convention."""
    with pytest.raises(ValueError):
        apply_member_conventions(_ensemble(np.array(["ctrl", "p1"])))


def test_keeps_scalar_member() -> None:
    """A single selected member stays a scalar ``member`` coordinate."""
    ds = _ensemble([3]).isel(number=0)
    ds = apply_member_conventions(ds)

    assert ds["member"].dims == ()
    assert int(ds["member"]) == 3
    assert ds["member"].attrs == MEMBER_ATTRS
    assert ds["t2m"].dims == ("reference_time", "grid_index")


def test_other_dim_name() -> None:
    """The source member dim can have another name, e.g. ``realization``."""
    ds = _ensemble([1, 2]).rename(number="realization")
    ds = apply_member_conventions(ds, dim="realization")

    assert ds["t2m"].dims[0] == "member"


def test_without_members_is_unchanged() -> None:
    """A dataset without a member coordinate is returned unchanged."""
    ds = _ensemble([1, 2]).drop_vars("number").rename(number="x")

    assert apply_member_conventions(ds).identical(ds)
