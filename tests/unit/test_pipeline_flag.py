"""Tests for the --skip-pipeline flag logic."""

import pytest

from tests.pipeline.flag import apply_skip_pipeline, PIPELINE_MARKER, should_skip_pipeline


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


class _FakeItem:
    """Minimal pytest.Item stand-in; ``keywords`` includes directory names like real items."""

    def __init__(self, *, marked: bool) -> None:
        self.keywords = {"pipeline": True}  # the tests/pipeline/ directory name
        self._marked = marked
        self.added: list[object] = []

    def get_closest_marker(self, name: str) -> object | None:
        return object() if self._marked and name == PIPELINE_MARKER else None

    def add_marker(self, marker: object) -> None:
        self.added.append(marker)


class _FakeConfig:
    def getoption(self, name: str) -> bool:
        assert name == "--skip-pipeline"
        return True


def test_skip_applies_only_to_marked_items_not_to_the_pipeline_directory() -> None:
    marked, unmarked = _FakeItem(marked=True), _FakeItem(marked=False)
    apply_skip_pipeline(config=_FakeConfig(), items=[marked, unmarked])  # type: ignore[arg-type]
    assert len(marked.added) == 1
    assert unmarked.added == []
