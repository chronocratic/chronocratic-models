"""Skip logic for slow end-to-end pipeline tests."""

from collections.abc import Mapping
import os

import pytest

PIPELINE_MARKER = "pipeline"
SKIP_ENV_VAR = "SKIP_PIPELINE_TESTS"


def should_skip_pipeline(*, flag: bool, environ: Mapping[str, str]) -> bool:
    """Decide whether pipeline tests are disabled.

    Args:
        flag: Value of the ``--skip-pipeline`` command-line option.
        environ: Environment mapping to read ``SKIP_PIPELINE_TESTS`` from.

    Returns:
        True if the flag is set or the env var is a non-empty value other than ``"0"``.
    """
    return flag or environ.get(SKIP_ENV_VAR, "0") not in ("", "0")


def apply_skip_pipeline(*, config: pytest.Config, items: list[pytest.Item]) -> None:
    """Add a skip marker to every ``pipeline``-marked item when disabled.

    Args:
        config: Active pytest config (provides the option value).
        items: Collected test items, mutated in place.
    """
    if not should_skip_pipeline(flag=config.getoption("--skip-pipeline"), environ=os.environ):
        return
    skip = pytest.mark.skip(reason="pipeline tests disabled (--skip-pipeline)")
    for item in items:
        if item.get_closest_marker(PIPELINE_MARKER) is not None:
            item.add_marker(skip)
