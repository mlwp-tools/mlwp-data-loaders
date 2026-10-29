---
status: accepted
date: 2026-10-29
---

# Trait-axis conventions are defined and checked in mlwp-data-specs

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

Where should these conventions be defined and enforced, so that the bundled loaders and loaders written by users produce the same coordinates?

## Decision Drivers

* **Datasets from different sources are compared and combined.** Typically a forecast is verified against point observations.
  * Checking whether two datasets share a grid means comparing latitude and longitude.
  * Interpolating to stations must work whatever longitude convention each source uses: anemoi global datasets store 0–360, observations use [-180, 180).
* **Time alignment works on time labels.** It intersects reference and lead times, selects observations at valid times, slices, checks that reference times are evenly spaced, and adds lead times to reference times.
  * This needs time labels that are sorted and unique.
  * Unsorted labels break slicing and the spacing check. Duplicate labels make `sel` return extra entries.
* **Users bring their own loaders, and need one place to look.** The specs are what a loader author reads and what validation checks. A second set of rules, in loader code, leaves them unsure whether following the specs is enough.
* **Loaders must stay lazy and keep their graphs small.** The rewrites in #24, #25 and #27 cut loading to reading coordinates only, and keep the number of dask layers independent of the number of files. Following the conventions must not read data variables or add dask layers to data that already follows them. The same goes for checking them.
* **The specs and xarray disagree on `lead_time` units.** The specs require a `units` attribute on `lead_time`. With a `timedelta64` `lead_time` that attribute makes `to_zarr` and `to_netcdf` raise, and xarray drops it when it reads a CF file back (mlwp-data-specs#39).

## Considered Options

1. **Public helpers in mlwp-data-loaders**, one per trait axis, which every loader calls last.
2. **Each loader sets its coordinates by hand**, as now.
3. **mxalign normalises the datasets** when it aligns them.
4. **Conventions in mlwp-data-specs, checked by the validator.** Each loader follows them however it likes.

## Decision Outcome

Chosen option: 4, because it gives the conventions a single definition, in the specs, and validation then enforces them for the bundled loaders and user loaders alike. mlwp-data-specs#40 adds the conventions below to the specs, with checks.

Public helpers (option 1) were the first proposal. They are rejected as the way to enforce the conventions. Next to the specs, they would be a second definition of the conventions, and someone writing a loader couldn't tell whether calling them is required, enough, or optional.

The bundled loaders may still share code to follow the conventions, as private helpers: one per trait axis in a private module (`mlwp_data_loaders._conventions`), called last by each loader. They are an implementation detail. They are not exported or documented for user loaders, and they are not what makes a dataset conform: validation is. Each helper:
* sets its axis's `mlwp_*_trait` attribute, as a plain string (mlwp-data-loaders#31);
* can rename its coordinates from the source names (e.g. `dim="values"`, `lead_time="step"`);
* only touches coordinates: it never reads data variables and leaves chunking to the loader;
* is idempotent, and the order of the calls doesn't matter.

The conventions, which go into the specs:

| axis | convention |
|---|---|
| all | Dimension order `(member \| quantile, time dims, other dims, space dim)`. The space dim comes last because it is never split and is read as one field. A transpose that changes nothing adds no dask layer. |
| space | Latitude and longitude have CF `standard_name` and `units` (already in the specs). A POINT `elevation` coordinate has `units: m` and `standard_name: surface_altitude`. |
| time | Time dimension coordinates are increasing and unique. The bundled loaders sort them if they are not increasing, and raise a `ValueError` on duplicate labels. |
| time | CF standard names: `forecast_reference_time`, `forecast_period` and `time`. `units` and `calendar` attributes are dropped from datetime coordinates. Until mlwp-data-specs#39 is settled, `lead_time` keeps `units: hours` as an attribute. |
| uncertainty | `member` has integer labels, the `realization` attributes and comes first (mlwp-data-specs#35). The trait is `ensemble` if `member` is a dimension and `deterministic` otherwise. There is no quantile convention yet. |

Left out of the conventions (decided 2026-10-29), although they appear in the table of differences above:
* the longitude range, because mxalign's space alignment is made independent of the longitude convention instead (mlwp-tools/mxalign#50), so datasets keep their source's longitudes, 0–360 or [-180, 180);
* the latitude/longitude dtype, because xarray upcasts float32 coordinates where needed;
* time dtypes and resolution, because since the loader rewrites the bundled loaders no longer disagree on them.

The bundled loaders only handle stacked layouts (`grid_index` or `point_index`). That is a limit of the private helpers, not a convention: the specs keep accepting `xc`/`yc` and regular latitude/longitude grids.

### Consequences

* Good, because each convention has one definition, in the specs, and validation catches loaders written by users too.
* Good, because a loader author has one thing to follow, the specs, and the validation report says what is wrong.
* Good, because the bundled loaders still share one implementation, so they can't drift apart, without adding public API to mlwp-data-loaders that must then stay stable.
* Good, because mlwp-data-loaders has no public helper for loader authors to wonder about. `apply_member_conventions`, public since #27, becomes private. It hasn't been released, so it needs no deprecation.
* Bad, because a user loader has to implement the conventions itself, e.g. sorting the time labels, instead of calling a helper.
* Bad, because until mlwp-data-specs#40 is done, the conventions beyond the current specs are only followed by the bundled loaders, not checked.
* Bad, because until mlwp-tools/mxalign#50 is done, aligning a 0–360 grid, such as an anemoi global dataset, with [-180, 180) observations silently gives NaN at the stations west of Greenwich.
* Bad, because the output of every bundled loader changes:
  * HARP's `altitude` is renamed `elevation`;
  * `reference_time` is sorted;
  * duplicate forecasts raise an error instead of being kept.
* Neutral, because the private space helper only handles stacked layouts (`grid_index`, `point_index`) and raises `NotImplementedError` for `latitude`/`longitude` and `xc`/`yc` grids. No bundled loader produces those yet, and since longitude isn't wrapped, supporting them needs no change to the data.
* Neutral, because the checks in the specs must stay cheap on lazy datasets. Attribute and dimension-order checks read no values. The sorted/unique checks read coordinate values only.
* Neutral, because until mlwp-data-specs#39 is settled, a dataset with a `lead_time` still can't be written without first dropping the `units` attribute. Moving it to `encoding` afterwards changes one constant.

### Confirmation

* The specs have a check for each convention (mlwp-data-specs#40).
* The tests of each private helper check, on synthetic datasets:
  * renames, attributes, sorting, the duplicate error, dimension order and idempotence;
  * that data already following the conventions gains no dask layer;
  * that no data variable is read.
* Each loader's tests check the conventions on its output and still pass `load_and_validate_dataset`. Once the specs check the conventions, that call covers them too.
* The anemoi-inference tests also check that the number of graph layers doesn't change.

## Pros and Cons of the Options

### Public helpers in mlwp-data-loaders

* Good, because there is one implementation, and its tests apply to all loaders that call it.
* Good, because a user loader gets the conventions with three calls instead of copying them from the bundled loaders.
* Bad, because the helpers become a second definition of the conventions next to the specs. A loader author can't tell whether calling them is required, enough, or optional.
* Bad, because it adds three public functions to mlwp-data-loaders, which must then stay stable.
* Bad, because the conventions beyond the specs are only enforced for loaders that call the helpers.

### Each loader sets its coordinates by hand

* Good, because there is no shared code to keep compatible.
* Bad, because it produced the differences in the table above.

### mxalign normalises the datasets

* Good, because mxalign sees every dataset, wherever it comes from.
* Bad, because the datasets are then valid by the specs but still need fixing, and every other consumer has to do the same.
* Bad, because mxalign would then sort and relabel data during alignment, where it is hard to tell a convention problem from a data problem.

### Conventions in mlwp-data-specs, checked by the validator

* Good, because the specs are where the conventions are defined, and validation catches user loaders too.
* Good, because loader authors have one place to look.
* Neutral, because every loader still writes its own code to follow the conventions. The bundled loaders share theirs as private helpers.
* Bad, because the specs need new checks before the conventions are enforced (mlwp-data-specs#40).

## More Information

* mlwp-data-specs#40: the conventions above, as requirements with checks.
* mlwp-tools/mxalign#50: space alignment that works with either longitude convention.
* mlwp-data-specs#35: member conventions. mlwp-data-specs#37: identifying the control member (open).
* mlwp-data-specs#39: `lead_time` units.
* mlwp-data-loaders#31: write the `mlwp_*_trait` attributes as plain strings.
* mlwp-data-loaders#24, #25, #27: the lazy loader rewrites. The private helpers and their use in all four loaders come in one PR, now that those three are merged. It changes their loaders and makes #27's `apply_member_conventions` private.
* Not yet done: regular latitude/longitude grids, quantile conventions, merging `cf` and `pf` members (waits for mlwp-data-specs#37).
