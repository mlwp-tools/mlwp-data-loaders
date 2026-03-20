"""CLI for loading datasets through loader modules and validating them."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from loguru import logger
from mlwp_data_specs import __version__ as specs_version
from mlwp_data_specs.api import validate_dataset

from .api import load_dataset


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        The configured argument parser.
    """
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
        help="Loader module or .py script defining the loader traits",
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
    """Run dataset loading and validation from CLI arguments.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Command line arguments. Defaults to None, which uses sys.argv[1:].

    Returns
    -------
    int
        Exit code: 0 for success, 1 for validation failures.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    storage_options = {}
    if args.s3_endpoint_url:
        storage_options["endpoint_url"] = args.s3_endpoint_url
    if args.s3_anon:
        storage_options["anon"] = True

    dataset_input = (
        args.dataset_paths[0] if len(args.dataset_paths) == 1 else args.dataset_paths
    )

    logger.info(f"Using mlwp-data-specs {specs_version}")

    # Load the dataset
    res = load_dataset(
        dataset_input,
        loader=args.loader,
        storage_options=storage_options or None,
        return_dataset_traits=True,
    )
    assert isinstance(res, tuple)
    ds, dataset_traits = res

    time_profile = dataset_traits.get("time_profile")
    space_profile = dataset_traits.get("space_profile")
    uncertainty_profile = dataset_traits.get("uncertainty_profile")

    report = validate_dataset(
        ds,
        time=time_profile,
        space=space_profile,
        uncertainty=uncertainty_profile,
    )

    report.console_print()
    return 1 if report.has_fails() else 0


if __name__ == "__main__":
    raise SystemExit(main())
