"""
Conventions for the ensemble ``member`` coordinate.

Loaders that return ensembles call :func:`apply_member_conventions` last, so
that every ensemble follows the ``mlwp-data-specs`` ensemble profile the same
way, whatever name and dtype the source gives its members.
"""

import numpy as np
import xarray as xr

MEMBER_ATTRS = {
    "standard_name": "realization",
    "units": "1",
    "long_name": "ensemble member number",
}


def apply_member_conventions(ds: xr.Dataset, dim: str = "number") -> xr.Dataset:
    """
    Rename the member coordinate of ``ds`` to ``member`` and apply its conventions.

    - ``dim`` is renamed to ``member``, both as a dimension and as a scalar
      coordinate (one member selected from an ensemble).
    - Numeric string labels (e.g. ``"0"``, ``"1"``) are cast to integers.
      Labels are never renumbered by position.
    - ``member`` gets the CF attributes of the ``realization`` standard name.
    - ``member`` becomes the first dimension of every data variable that has it.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset whose member coordinate is ``dim`` (or already ``member``).
    dim : str, default: "number"
        Name of the member coordinate in ``ds``; cfgrib calls it ``number``.

    Returns
    -------
    xr.Dataset
        ``ds`` with the member conventions applied, or ``ds`` unchanged if it
        has no ``dim`` coordinate.
    """
    if dim != "member" and (dim in ds.variables or dim in ds.dims):
        ds = ds.rename({dim: "member"})
    if "member" not in ds.coords:
        return ds

    member = ds["member"]
    values = member.values
    if values.dtype.kind in "OSU":
        values = values.astype(np.int64)
    ds = ds.assign_coords(
        member=(member.dims, values, {**member.attrs, **MEMBER_ATTRS})
    )
    if "member" in ds.dims:
        ds = ds.transpose("member", ...)
    return ds
