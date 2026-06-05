"""Intake driver that loads datasets via mlwp-data-loaders loader modules."""

from __future__ import annotations

from typing import Any

import xarray as xr
from intake.source.base import DataSource, Schema

from mlwp_data_loaders.core import get_loader_func


class MLWPLoaderDriver(DataSource):
    container = "xarray"
    name = "mlwp_loader"
    version = "0.1.0"
    partition_access = False

    def __init__(
        self,
        dataset_path: str | list[str],
        loader: str,
        chunks: str | dict | None = None,
        variables: str | list[str] | None = None,
        storage_options: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self._dataset_path = dataset_path
        self._loader_str = loader
        self._chunks = chunks
        self._variables = variables
        self._storage_options = storage_options or {}
        self._kwargs = kwargs
        self._ds: xr.Dataset | None = None
        super().__init__(metadata=metadata)

    def _load(self) -> xr.Dataset:
        if self._ds is None:
            loader_func = get_loader_func(self._loader_str)
            loader_kwargs: dict[str, Any] = dict(self._kwargs)
            loader_kwargs["chunks"] = self._chunks
            if self._variables is not None:
                loader_kwargs["variables"] = self._variables
            if self._storage_options:
                loader_kwargs["storage_options"] = self._storage_options
            self._ds = loader_func(self._dataset_path, **loader_kwargs)
        return self._ds

    def _get_schema(self) -> Schema:
        ds = self._load()
        data_vars = {k: str(v.dtype) for k, v in ds.data_vars.items()}
        return Schema(
            datashape=dict(ds.sizes),
            dtype=data_vars,
            shape=None,
            npartitions=1,
            metadata=dict(ds.attrs),
            extra_metadata={},
        )

    def read(self, **kwargs: Any) -> xr.Dataset:
        return self._load()

    def to_dask(self, **kwargs: Any) -> xr.Dataset:
        return self._load()

    def _close(self) -> None:
        self._ds = None
        self._schema = None
