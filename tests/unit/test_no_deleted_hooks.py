"""Verify no deleted hooks remain in source code after 2-hook refactor (D-02).

Regression test: ensures _get_encoder_module, _prepare_inputs, and _postprocess
are not defined in any model class dictionary (only comments are allowed).
"""

from __future__ import annotations

import ast
import pathlib

import pytest

# Models that use BasicEncodingMixin
MODEL_FILES = [
    # 5 simple models (already refactored in 09-02)
    pathlib.Path("src/chronocratic/models/generative/timevae/model.py"),
    pathlib.Path("src/chronocratic/models/recurrent/timenet/model.py"),
    pathlib.Path("src/chronocratic/models/recurrent/recurrentae/model.py"),
    pathlib.Path("src/chronocratic/models/convolutional/standard/mcl/model.py"),
    pathlib.Path("src/chronocratic/models/convolutional/standard/tstcc/model.py"),
    # 2 complex models (refactored in 09-04)
    pathlib.Path("src/chronocratic/models/transformer/tst/model.py"),
    pathlib.Path("src/chronocratic/models/convolutional/standard/series2vec/model.py"),
]

DELETED_HOOKS = {"_get_encoder_module", "_prepare_inputs", "_postprocess"}


@pytest.mark.parametrize("model_file", MODEL_FILES)
def test_no_deleted_hooks_in_model(model_file: pathlib.Path) -> None:
    """No model class defines deleted hooks (_get_encoder_module, _prepare_inputs, _postprocess)."""
    root = pathlib.Path(__file__).resolve().parent.parent.parent
    full_path = root / model_file
    source = full_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(full_path))

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = {n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            found = methods & DELETED_HOOKS
            assert not found, (
                f"{node.name} in {model_file} still defines deleted hooks: {found}"
            )
