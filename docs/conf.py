"""Sphinx configuration for chronocratic-models documentation."""

from __future__ import annotations

from pathlib import Path
import sys

# Insert project root source directory at the front, so that Sphinx can
# import the chronocratic.models package even when docs are built from a
# clean environment that does not have the package installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Import must follow the sys.path modification so that chronocratic.models
# is discoverable. The ``__version__`` value is written dynamically by
# setuptools_scm (see pyproject.toml [tool.setuptools_scm]).
from chronocratic.models import __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-information

project: str = "chronocratic-models"
html_title: str = "chronocratic-models"
copyright: str = "2026-Present, The Chronocratic Developers"  # noqa: A001
author: str = "The Chronocratic Developers"
release: str = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#configuration

extensions: list[str] = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "myst_parser"]

# Napoleon (Google-style docstring parsing)
napoleon_use_google_style: bool = True

# Autodoc
autodoc_default_options: dict[str, str] = {"member-order": "bysource"}

# MyST parser
myst_enable_extensions: list[str] = ["colon_fence", "deflist", "substitution"]

# Support both .rst and .md source files
source_suffix: dict[str, str] = {".rst": "restructuredtext", ".md": "markdown"}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme: str = "pydata_sphinx_theme"
html_theme_options: dict = {
    "navigation_depth": 3,
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
}

html_static_path: list[str] = ["_static"]
html_css_files: list[str] = ["custom.css"]

# Suppress non-critical warnings
suppress_warnings: list[str] = ["efifo"]
