# Developing

## Environment

This project uses `uv` for dependency management and local commands.

Install dependencies into the project environment with:

```bash
uv sync --extra test --group dev
```

If you only need to run a one-off command, you can also use `uv run ...`
without activating the environment.

## Running Tests

Run the full test suite with:

```bash
uv run python -m pytest
```

Run a single test file with:

```bash
uv run python -m pytest tests/test_anemoi_datasets_integration.py
```

## Pre-commit

This repository includes a checked-in
[`.pre-commit-config.yaml`](.pre-commit-config.yaml).
CI runs the same hooks via `.github/workflows/pre-commit.yml`.

Install the development dependencies first:

```bash
uv sync --extra test --group dev
```

Then run the hooks locally with:

```bash
pre-commit run --all-files
```

You can also install the git hook so checks run before each commit:

```bash
pre-commit install
```

## CA Certificates

`certifi` is included so that `botocore` and `aiobotocore` use an up-to-date CA
bundle when opening datasets from custom S3 endpoints over HTTPS.

This matters for the ECMWF object-store endpoint used by the
`anemoi-datasets` integration test:

- the endpoint certificate chain is valid for standard HTTPS clients
- older bundled CA bundles in `botocore` do not include the required
  `HARICA TLS RSA Root CA 2021` root
- when `certifi` is installed, `botocore` uses the `certifi` bundle by default

The minimum supported version is `certifi>=2021.10.8` because that is the first
`certifi` release we verified to include `HARICA TLS RSA Root CA 2021`.
