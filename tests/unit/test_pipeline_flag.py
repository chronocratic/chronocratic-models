"""Tests for the --skip-pipeline flag logic."""

import pytest

from tests.pipeline.flag import PIPELINE_MARKER, should_skip_pipeline


@pytest.mark.parametrize(
    ("flag", "environ", "expected"),
    [
        (False, {}, False),
        (True, {}, True),
        (False, {"SKIP_PIPELINE_TESTS": "1"}, True),
        (False, {"SKIP_PIPELINE_TESTS": "0"}, False),
        (False, {"SKIP_PIPELINE_TESTS": ""}, False),
    ],
)
def test_should_skip_pipeline(flag: bool, environ: dict[str, str], expected: bool) -> None:
    assert should_skip_pipeline(flag=flag, environ=environ) is expected


def test_marker_name_is_registered(pytestconfig: pytest.Config) -> None:
    registered = pytestconfig.getini("markers")
    assert any(line.startswith(PIPELINE_MARKER) for line in registered)
