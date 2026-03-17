"""CLI for loading datasets through loader modules and validating them."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from loguru import logger

from mlwp_data_specs import validate_dataset
from mlwp_data_specs import __version__ as specs_version
from mlwp_data_specs.specs.traits.spatial_coordinate import Space
from mlwp_data_specs.specs.traits.time_coordinate import Time
from mlwp_data_specs.specs.traits.uncertainty import Uncertainty

from .api import load_dataset
from .mxalign_api import validate_dataset_with_mxalign


def _choice_values(enum_cls) -> list[str]:
    """List sorted enum values for CLI choices."""
    return sorted(item.value for item in enum_cls)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Load a dataset through a loader module and validate it with "
            "mlwp-data-specs."
        )
    )
    parser.add_argument(
        "dataset_paths",
        nargs="+",
        help="One or more dataset paths/URLs to load and validate",
    )
    parser.add_argument(
        "--loader",
        required=True,
        help="Loader module or .py script defining the loader hooks",
    )
    parser.add_argument(
        "--space", choices=_choice_values(Space), help="Space trait name"
    )
    parser.add_argument("--time", choices=_choice_values(Time), help="Time trait name")
    parser.add_argument(
        "--uncertainty",
        choices=_choice_values(Uncertainty),
        help="Uncertainty trait name",
    )
    parser.add_argument(
        "--s3-endpoint-url",
        default=None,
        help="Optional S3 endpoint URL for opening the dataset",
    )
    parser.add_argument(
        "--s3-anon", action="store_true", help="Use anonymous S3 access"
    )
    return parser


@logger.catch
def main(argv: Sequence[str] | None = None) -> int:
    """Run dataset loading and validation from CLI arguments."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not any([args.space, args.time, args.uncertainty]):
        parser.error(
            "At least one trait must be selected with --space/--time/--uncertainty"
        )

    storage_options = {}
    if args.s3_endpoint_url:
        storage_options["endpoint_url"] = args.s3_endpoint_url
    if args.s3_anon:
        storage_options["anon"] = True

    dataset_input = args.dataset_paths[0] if len(args.dataset_paths) == 1 else args.dataset_paths

    logger.info(f"Using mlwp-data-specs {specs_version}")
    ds = load_dataset(
        dataset_input,
        loader=args.loader,
        time=args.time,
        space=args.space,
        uncertainty=args.uncertainty,
        storage_options=storage_options or None,
    )
    report = validate_dataset(
        ds,
        time=args.time,
        space=args.space,
        uncertainty=args.uncertainty,
    )
    report += validate_dataset_with_mxalign(
        ds,
        time=args.time,
        space=args.space,
        uncertainty=args.uncertainty,
    )
    report.console_print()
    return 1 if report.has_fails() else 0


if __name__ == "__main__":
    raise SystemExit(main())
