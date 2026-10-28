---
status: proposed
date: 2026-10-28
---

# One conventions helper per trait axis, called last by every loader

## Context and Problem Statement

Every loader writes its own code for the space, time and uncertainty coordinates: renames, attributes, dimension order and the three `mlwp_*_trait` attributes. That is 15–20 lines per loader. mlwp-data-specs validates the result, but it only checks names and a few attributes. The bundled loaders therefore produce datasets that pass validation and still disagree:

| | anemoi-datasets | anemoi-inference | IFS forecast | HARP obstable |
|---|---|---|---|---|
| longitude range | as stored (0–360) | as stored | [-180, 180) | as stored (±180 for SYNOP) |
| latitude/longitude dtype | float32 | as stored | float64 | float64 |
| time resolution | ns | ns | ns | us (pandas 3 `read_sql`) |
| `reference_time` order | n/a | order of the paths | order of the paths | n/a |
| duplicate times | n/a | kept | kept | n/a |
| station elevation | n/a | n/a | n/a | `altitude` (the POINT spec names `elevation`) |

The anemoi-datasets, anemoi-inference and IFS columns describe those loaders as rewritten in mlwp-data-loaders#24, #25 and #27.

The `member` coordinate had the same problem. mlwp-data-specs#35 settled its conventions, and `apply_member_conventions` (mlwp-data-loaders#27) applies them in one place.

How should loaders make their datasets follow one convention on every trait axis, including the conventions the specs don't check?

## Decision Drivers

* **Datasets from different sources are compared and combined.** This drives the space conventions. Typically a forecast is verified against point observations.
  * Checking whether two datasets share a grid means comparing latitude and longitude exactly.
  * Interpolating to stations needs both datasets on the same longitude convention, and so does selecting a longitude range such as -10 to 30.
  * Observations use [-180, 180).
* **Time alignment works on time labels.** It intersects reference and lead times, selects observations at valid times, slices, checks that reference times are evenly spaced, and adds lead times to reference times.
  * This needs `datetime64` and `timedelta64` values that are sorted, unique and at one resolution.
  * Unsorted labels break slicing and the spacing check. Duplicate labels make `sel` return extra entries.
* **Loaders must stay lazy and keep their graphs small.** The rewrites in #24, #25 and #27 cut loading to reading coordinates only, and keep the number of dask layers independent of the number of files. A helper must not read data variables or add dask layers to data that already follows the conventions.
* **Users bring their own loaders.** A rule written down only in each bundled loader is easy for them to miss.
* **The specs and xarray disagree on `lead_time` units.** The specs require a `units` attribute on `lead_time`. With a `timedelta64` `lead_time` that attribute makes `to_zarr` and `to_netcdf` raise, and xarray drops it when it reads a CF file back (mlwp-data-specs#39).

## Considered Options

1. **One helper per trait axis in mlwp-data-loaders**, which every loader calls last.
2. **Each loader sets its coordinates by hand**, as now.
3. **mxalign normalises the datasets** when it aligns them.
4. **Rules in mlwp-data-specs only**, with no shared code.

## Decision Outcome

Chosen option: 1, because it is the only option that gives every bundled loader the same output from a single implementation, and user loaders can call it too. The conventions it applies are also proposed to the specs (mlwp-data-specs#39), so that option 4 follows later.

The package `mlwp_data_loaders.conventions` has one helper per trait axis:

```python
ds = apply_time_conventions(ds, Time.FORECAST)
ds = apply_space_conventions(ds, Space.GRID)
ds = apply_uncertainty_conventions(ds)
```

Each helper:
* sets its axis's `mlwp_*_trait` attribute;
* can rename its coordinates from the source names (e.g. `dim="values"`, `lead_time="step"`);
* only touches coordinates: it never reads data variables and leaves chunking to the loader;
* is idempotent, and the order of the calls doesn't matter.

The conventions:

| axis | convention |
|---|---|
| all | Dimension order `(member \| quantile, time dims, other dims, space dim)`. The space dim comes last because it is never split and is read as one field. A transpose that changes nothing adds no dask layer. |
| space | Stacked layouts only: `grid_index` or `point_index`. Longitude is wrapped to [-180, 180) by shifting only the values outside that range by one turn. Unlike `(lon + 180) % 360 - 180`, this leaves in-range values bit for bit, so grids from different sources still compare equal, and the shift is exact. It changes values, not the order of the data. `valid_min`, `valid_max` and `valid_range` are dropped from the longitude attributes. |
| space | Latitude and longitude are float64 numpy coordinates with CF `standard_name` and `units`. A POINT `elevation` coordinate gets `units: m` and `standard_name: surface_altitude`. |
| time | `reference_time` and `valid_time` are `datetime64[ns]`; `lead_time` is `timedelta64[ns]`. A numeric `lead_time` with `units` in s/seconds/h/hours is converted. A scalar time coordinate the profile requires becomes a dimension of length one. |
| time | Time dimensions are sorted if they are not increasing. Duplicate labels raise a `ValueError`. |
| time | CF standard names: `forecast_reference_time`, `forecast_period` and `time`. `units` and `calendar` attributes are dropped from datetime coordinates. Until mlwp-data-specs#39 is settled, `lead_time` keeps `units: hours` as an attribute. |
| uncertainty | `member` has integer labels, the `realization` attributes and comes first (mlwp-data-specs#35). The trait is `ensemble` if `member` is a dimension and `deterministic` otherwise. There is no quantile convention yet. |

### Consequences

* Good, because every bundled loader gives the same coordinates, so grid identity checks, interpolation to stations and time alignment don't need to handle each source differently.
* Good, because a user loader gets the conventions with three calls instead of copying them from the bundled loaders.
* Good, because the spec-checked parts (names, attributes, member order) can't drift between loaders.
* Bad, because the output of every bundled loader changes:
  * anemoi-datasets longitudes move from 0–360 to [-180, 180), and its latitude and longitude go from float32 to float64;
  * HARP's `altitude` is renamed `elevation`;
  * `reference_time` is sorted;
  * duplicate forecasts raise an error instead of being kept.
* Bad, because datasets on a regular latitude/longitude grid (dimensions `latitude`/`longitude` or `xc`/`yc`) are not handled yet. Wrapping their longitude means reordering the data. The helper raises `NotImplementedError` for them.
* Neutral, because until mlwp-data-specs#39 is settled, a dataset with a `lead_time` still can't be written without first dropping the `units` attribute. Moving it to `encoding` afterwards changes one constant.

### Confirmation

* The tests of each helper check, on synthetic datasets:
  * renames, attributes, dtypes, longitude wrapping, sorting, the duplicate error, dimension order and idempotence;
  * that data already following the conventions gains no dask layer;
  * that no data variable is read.
* Each loader's tests check the conventions on its output and still pass `load_and_validate_dataset`.
* The anemoi-inference tests also check that the number of graph layers doesn't change.

## Pros and Cons of the Options

### One helper per trait axis

* Good, because there is one implementation, and its tests apply to all loaders.
* Good, because it mirrors the specs' trait axes. Each helper is small and can be used on its own, e.g. by a user loader that already follows the time conventions.
* Neutral, because it adds three public functions to mlwp-data-loaders, which must then stay stable.
* Bad, because the conventions beyond the specs are, for now, only enforced for loaders that call the helpers.

### Each loader sets its coordinates by hand

* Good, because there is no shared code to keep compatible.
* Bad, because it produced the differences in the table above.

### mxalign normalises the datasets

* Good, because mxalign sees every dataset, wherever it comes from.
* Bad, because the datasets are then valid by the specs but still need fixing, and every other consumer has to do the same.
* Bad, because mxalign would then sort and relabel data during alignment, where it is hard to tell a convention problem from a data problem.

### Rules in mlwp-data-specs only

* Good, because the specs are where the conventions are defined, and validation catches user loaders too.
* Bad, because every loader would still write its own code to follow them.
* Neutral, because it complements option 1 rather than replacing it, and is proposed in mlwp-data-specs#39.

## More Information

* mlwp-data-specs#35: member conventions. mlwp-data-specs#37: identifying the control member (open).
* mlwp-data-specs#39: `lead_time` units and the conventions proposed for the specs.
* mlwp-data-loaders#24, #25, #27: the lazy loader rewrites. The helpers and their use in all four loaders come in one PR, opened after those three are merged, because it changes their loaders and renames #27's `apply_member_conventions` to `apply_uncertainty_conventions`.
* Not yet done: regular latitude/longitude grids, quantile conventions, merging `cf` and `pf` members (waits for mlwp-data-specs#37).
