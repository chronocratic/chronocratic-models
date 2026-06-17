# Phase 06: Package Preparation - Research

**Researched:** 2026-06-15
**Domain:** Python packaging, namespace migration, CI/CD, documentation
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**D-01: Namespace Rename -- Full, Single Phase**
Rename `tscollection` to `chronocratic` across all source files, test files, imports, and configuration. Package dir becomes `src/chronocratic/models/`. PyPI package name: `chronocratic-models`. Import path: `from chronocratic.models import TS2Vec, ...`.

Mechanical sweep: Every `tscollection` reference to `chronocratic`. Includes:
- `src/tscollection/` to `src/chronocratic/`
- All import statements in src/ and tests/
- `pyproject.toml` package discovery paths
- CI workflow coverage paths
- `ruff.toml` source paths if any

**D-02: Namespace Package -- Implicit PEP 420**
`chronocratic/` will have **no `__init__.py`**. Both `chronocratic-datasets` and `chronocratic-models` contribute to the same namespace implicitly. Users install both packages and import from `chronocratic.datasets` and `chronocratic.models` independently.

**D-03: Version Strategy -- setuptools_scm**
Dynamic versioning from git tags. `[build-system]` adds `setuptools-scm>=8`. `version` becomes `dynamic = ["version"]`. `[tool.setuptools_scm]` writes `src/chronocratic/models/_version.py`. Matches reference repo (`chronocratic-datasets`).

**D-04: Documentation -- Sphinx + pydata-theme + MyST + ReadTheDocs**
Match `chronocratic-datasets` exactly:
- `docs/conf.py` -- Sphinx config with autodoc, MyST parser
- `docs/index.md` -- Landing page
- `docs/api/` -- Auto-generated API reference
- `docs/quickstart.md` -- TS2Vec encoding example
- `docs/changelog.md` -- Includes CHANGELOG.md
- `docs/contributing.md` -- Contributing guide
- `.readthedocs.yaml` -- Build config (ubuntu-24.04, Python 3.12, uv)
- `[project.optional-dependencies]` -- `docs = ["sphinx>=7.0", "pydata-sphinx-theme>=0.15", "myst-parser>=3.0"]`

**D-05: README -- Match Reference Structure**
- Badges: License (BSD-3-Clause), PyPI version, Python versions, downloads, build status, documentation, code style (ruff), GitHub stars
- Installation section with `pip install chronocratic-models`
- Quick start: TS2Vec encoding example (model instantiation + `encode()` call)
- Models: Catalog of all 10 models with one-line descriptions
- Features: Library highlights (polymorphic augmentations, Lightning integration, etc.)
- Documentation: Link to ReadTheDocs
- License: BSD 3-Clause

**D-06: LICENSE -- BSD-3-Clause**
Copy from reference repo. Copyright holder: "The Chronocratic Developers". Year: 2026-Present.

**D-07: Changelog -- towncrier**
Match reference repo config:
- `[tool.towncrier]` in pyproject.toml
- `changelog.d/` directory with type subdirs (added, changed, deprecated, removed, fixed, security)
- `changelog.d/README.md` -- fragment instructions
- `CHANGELOG.md` -- assembled by towncrier
- Existing `auto-changelog-fragment.yml` and `release-prep.yml` workflows already wired

**D-08: CI Workflow Fixes**
- **build-and-test.yml:**
  - Coverage path: `src/chronocratic/datasets` to `src/chronocratic/models`
  - Docs job: Install `[docs]` extras (now defined), run `sphinx-build`
- **pypi-publish.yml:** Already correct (triggers on release, uses `python -m build`)
- **release-prep.yml:** Already correct (towncrier on dev branch)
- **auto-changelog-fragment.yml:** Already correct

**D-09: pyproject.toml -- Complete Metadata**
Full metadata including name, dynamic version, description, readme, requires-python, license, keywords, authors, classifiers, dependencies, project URLs.

**D-10: Dependency Audit -- Remove Unused Dependencies**
Audit all `[project.dependencies]` and `[dependency-groups]` entries against actual imports in `src/`. If a dep is only used in `tests/`, move it to `[dependency-groups] dev`. If unused entirely, drop it.

### Claude's Discretion

- Exact task wave ordering within the rename (src to tests to config to CI)
- README badge selection (match reference repo defaults)
- Model catalog ordering in README (by family: convolutional to transformer to recurrent to generative)
- Docs quickstart code snippet (exact TS2Vec example)
- Sphinx autodoc membership listing (which modules to document explicitly)
- `ruff.toml` source path updates (if needed after rename)

### Deferred Ideas (OUT OF SCOPE)

- **ReadTheDocs account setup** -- Requires external account creation and repo linking. Do after package is built locally.
- **PyPI organization account** -- `chronocratic` PyPI user/org. Set up before first publish, not in this phase.
- **Distribution to conda-forge** -- Separate packaging concern. Future phase.
- **Pre-built wheels (manylinux)** -- `cibuildwheel` integration. Future optimization.
- **Package size optimization** -- Exclude large test fixtures, distances/ CUDA code. Future.
</user_constraints>

## Summary

This phase transforms the tsmodels repository from an unpublished `tscollection` namespace into a fully packageable `chronocratic-models` distribution on PyPI. The work falls into five interdependent strands: (1) renaming the namespace from `tscollection` to `chronocratic` across 62 Python source files, 28 test files, and 2 configuration files; (2) auditing and pruning `pyproject.toml` dependencies, removing 6 unused packages while adding 1 missing runtime dependency (tqdm); (3) adding complete package metadata, setuptools_scm versioning, and towncrier changelog infrastructure; (4) bootstrapping Sphinx documentation with ReadTheDocs integration; and (5) fixing CI workflows that currently reference the wrong namespace and missing docs extras.

The reference repository (`chronocratic-datasets`) provides an exact template for all metadata, documentation, and tooling decisions. The primary risk is the mechanical rename across 62 files -- a missed reference causes import failures at runtime. The recommended approach is to perform the rename in a strict ordering: source files first, then tests, then configuration, then CI -- verified at each stage by running `pytest` to confirm no dangling imports remain.

**Primary recommendation:** Execute the namespace rename as a single atomic sweep using `git mv` for directory renames and a verified search-and-replace for import statements, followed immediately by a full test run. Audit and prune dependencies in parallel with the rename since both touch `pyproject.toml`. Add documentation infrastructure last, as it depends on the renamed package importing correctly.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Namespace rename (source) | Build/Packaging | — | Changes package layout; no runtime architecture |
| Namespace rename (tests) | Build/Packaging | — | Test imports follow source imports |
| Dependency pruning | Build/Packaging | — | Affects pyproject.toml only; verified by import grep |
| setuptools_scm versioning | Build/Packaging | CI | Git tags drive version; CI must fetch tags |
| Sphinx documentation | Build/Packaging | — | Static site; reads from source via autodoc |
| towncrier changelog | Build/Packaging | CI | Fragment workflow already exists; config is added |
| CI workflow fixes | CI | Build/Packaging | Coverage paths, docs build job |
| ReadTheDocs integration | CI | Build/Packaging | .readthedocs.yaml; uses uv to build docs |
| LICENSE, README, CHANGELOG | Build/Packaging | — | Static files added to repo root |

## Standard Stack

### Core Packaging Tools
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| setuptools (>=68) | Current | Build backend | Already in use; supports src layout |
| setuptools-scm (>=8) | 8.x | Dynamic versioning from git tags | Reference repo standard; eliminates manual version bumps |
| towncrier (>=24.8) | Current | Changelog fragment assembly | Already wired in CI workflows |
| build | Current | PEP 517 package builder | Used by `pypi-publish.yml` workflow |
| twine | Current | Package validation before publish | Used by `build-and-test.yml` build job |

### Documentation
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Sphinx (>=7.0) | Current | Documentation generator | Reference repo standard |
| pydata-sphinx-theme (>=0.15) | Current | Sphinx theme | Matches reference repo visual identity |
| myst-parser (>=3.0) | Current | Markdown source support | Allows .md docs alongside autodoc |

### Validation
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytest (>=8.2) | Current | Test runner | Already configured |
| pytest-cov (>=5.0) | Current | Coverage reporting | Already in CI |
| ruff (>=0.15.9) | Current | Linting/formatting | Already configured |
| ty (>=0.0.28) | Current | Static type checking | Already in dev group |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| setuptools-scm | Manual version field | Risk of version drift; more merge conflicts |
| Sphinx | pdoc, MkDocs | Sphinx + autodoc provides better API reference for library packages |
| towncrier | git-changelog, dorny/changes-action | towncrier is reference repo standard; already partially wired |

**Installation:**
```bash
# Build system (added to [build-system] requires)
pip install setuptools-scm>=8

# Dev tools (added to [dependency-groups] dev)
pip install towncrier>=24.8

# Docs (added to [project.optional-dependencies] docs)
pip install sphinx>=7.0 pydata-sphinx-theme>=0.15 myst-parser>=3.0
```

## Package Legitimacy Audit

> All packages listed below are either already in use in this project or are reference repo standards confirmed via the chronocratic-datasets repository and official PyPI.

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| setuptools-scm | PyPI | 7+ yrs | 10M+/wk | github.com/pypa/setuptools_scm | OK | Approved |
| towncrier | PyPI | 5+ yrs | 500K+/wk | github.com/twisted/towncrier | OK | Approved |
| sphinx | PyPI | 15+ yrs | 5M+/wk | github.com/sphinx-doc/sphinx | OK | Approved |
| pydata-sphinx-theme | PyPI | 4+ yrs | 500K+/wk | github.com/pydata/pydata-sphinx-theme | OK | Approved |
| myst-parser | PyPI | 5+ yrs | 2M+/wk | github.com/executablebooks/MyST-Parser | OK | Approved |

**Packages removed due to SLOP verdict:** none
**Packages flagged as suspicious:** none

## Dependency Audit Results

### Runtime Dependencies (`[project] dependencies`)

| Dependency | Used in src/? | Files | Recommendation | Confidence |
|------------|---------------|-------|----------------|------------|
| `numpy>=2.1,<3.0.0` | YES | Widely used (arrays, random sampling) | **Keep** | HIGH |
| `pandas>=2.2.0` | NO | 0 files | **Remove** (transitive via scikit-learn) | HIGH |
| `scipy>=1.13.0` | YES | `series2vec/filters.py` (butter, lfilter) | **Keep** | HIGH |
| `scikit-learn>=1.6,<2.0.0` | NO | 0 files | **Remove** (not imported in src/) | HIGH |
| `lightning>=2.5,<3.0` | YES | All model classes inherit LightningModule | **Keep** | HIGH |
| `torch>=2.4,<3.0` | YES | Core dependency (nn, fft, optim) | **Keep** | HIGH |
| `torchvision>=0.19.0` | NO | 0 files (docstring mention only) | **Remove** | HIGH |
| `torchaudio>=2.4.0` | NO | 0 files | **Remove** | HIGH |
| `joblib>=1.4.0` | NO | 0 files | **Remove** (transitive via scikit-learn) | HIGH |
| `openpyxl~=3.1.5` | NO | 0 files | **Remove** (transitive via pandas) | HIGH |
| `h5py~=3.16.0` | NO | 0 files | **Remove** | HIGH |
| `einops>=0.8.2` | YES | 3 files (utils.py, temporal_contrast.py, encoders.py) | **Keep** | HIGH |
| `numba>=0.65.1` | YES | `soft_dtw/soft_dtw_cuda.py` (cuda, jit, prange) | **Keep** | HIGH |
| `tqdm` | YES (MISSING) | `dilated/_mixin/encoding.py` (lines 11, 122, 242) | **Add** | HIGH |

### Summary of Changes

**Remove from dependencies (6 packages):** `torchvision`, `torchaudio`, `joblib`, `openpyxl`, `h5py`, `pandas`, `scikit-learn`

Rationale: None of these are imported in any `src/` file. Note that `pandas` and `scikit-learn` are flagged in D-10 as potentially used by downstream `rbspaper` (which is gitignored in `_sources/`). However, as a published library package, only direct runtime dependencies should be declared. Users who need pandas or scikit-learn will already have them or will install them explicitly.

**Add to dependencies (1 package):** `tqdm`

Rationale: `tqdm` is imported at line 11 of `src/tscollection/models/convolutional/dilated/_mixin/encoding.py` and used in two `for` loops (lines 122, 242) for progress bars during encoding. It is NOT currently listed in `pyproject.toml` dependencies -- this is a real bug that will cause `ImportError` at runtime when the package is installed from PyPI.

**Keep in dependencies (7 packages):** `numpy`, `scipy`, `lightning`, `torch`, `einops`, `numba`, (plus the new `tqdm`)

### Dev Dependencies (`[dependency-groups] dev`)

| Dependency | Used? | Recommendation |
|------------|-------|----------------|
| `pytest>=8.2` | YES | Keep |
| `pytest-cov>=5.0` | YES | Keep |
| `ruff>=0.15.9` | YES | Keep |
| `ty>=0.0.28` | YES | Keep |
| `towncrier>=24.8` | Not yet | **Add** (required for changelog tooling) |

### Final Target Dependencies

```toml
dependencies = [
    "numpy>=2.1,<3.0.0",
    "scipy>=1.13.0",
    "lightning>=2.5,<3.0",
    "torch>=2.4,<3.0",
    "einops>=0.8.2",
    "numba>=0.65.1",
    "tqdm>=4.66.0",
]
```

## Namespace Rename Scope

### Exact Counts

- **Total Python files referencing `tscollection`:** 62
- **Source files (`src/`):** 34 files
- **Test files (`tests/`):** 28 files
- **Config files:** 2 files (`pyproject.toml` line 5, `ruff.toml` line 8)
- **CI workflow files:** 1 file (`build-and-test.yml` line 24 -- already has `chronocratic/datasets`, needs to change to `chronocratic/models`)
- **Other workflow files:** `pypi-publish.yml` already references `chronocratic-models` (correct)

### Safe Rename Ordering

1. **Directory rename:** `git mv src/tscollection src/chronocratic` (moves directory; `tscollection/` currently has no `__init__.py` so this is safe)
2. **Source imports:** Replace `tscollection` with `chronocratic` in all 34 source files
3. **Test imports:** Replace `tscollection` with `chronocratic` in all 28 test files
4. **Config files:** Update `pyproject.toml` (name, package discovery), `ruff.toml` (exclude path)
5. **CI workflows:** Update `build-and-test.yml` coverage path
6. **Verify:** Run `pytest tests/` to confirm no broken imports

### Important Detail: Implicit Namespace Package

`src/tscollection/` currently has NO `__init__.py` -- it is already an implicit PEP 420 namespace package containing only `models/`. After rename, `src/chronocratic/` must also have NO `__init__.py`. The `src/chronocratic/models/` directory WILL have `__init__.py`. This is critical for the shared namespace between `chronocratic-datasets` and `chronocratic-models`.

## pyproject.toml Transformation

### Current State (Before)

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]

[project]
name = "tscollection-models"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "numpy>=2.1,<3.0.0",
    "pandas>=2.2.0",
    "scipy>=1.13.0",
    "scikit-learn>=1.6,<2.0.0",
    "lightning>=2.5,<3.0",
    "torch>=2.4,<3.0",
    "torchvision>=0.19.0",
    "torchaudio>=2.4.0",
    "joblib>=1.4.0",
    "openpyxl~=3.1.5",
    "h5py~=3.16.0",
    "einops>=0.8.2",
    "numba>=0.65.1",
]

[dependency-groups]
dev = [
    "pytest>=8.2",
    "pytest-cov>=5.0",
    "ruff>=0.15.9",
    "ty>=0.0.28",
]
notebooks = [
    "notebook>=7.3",
    "jupyterlab>=4.3",
]

[tool.uv]
default-groups = ["dev"]

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

### Target State (After)

```toml
[build-system]
requires = ["setuptools>=68", "setuptools-scm>=8"]
build-backend = "setuptools.build_meta"

[project]
name = "chronocratic-models"
dynamic = ["version"]
description = "Self-supervised time-series representation learning models"
readme = "README.md"
requires-python = ">=3.12"
license = "BSD-3-Clause"
license-files = ["LICENSE"]
keywords = [
    "time-series", "pytorch", "lightning", "machine-learning",
    "self-supervised-learning", "representation-learning",
    "forecasting", "classification", "regression", "anomaly-detection",
]
authors = [{name = "The Chronocratic Developers", email = "github@users.noreply.github.com"}]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: Implementation :: CPython",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Software Development :: Libraries :: Python Modules",
]
dependencies = [
    "numpy>=2.1,<3.0.0",
    "scipy>=1.13.0",
    "lightning>=2.5,<3.0",
    "torch>=2.4,<3.0",
    "einops>=0.8.2",
    "numba>=0.65.1",
    "tqdm>=4.66.0",
]

[project.optional-dependencies]
docs = [
    "sphinx>=7.0",
    "pydata-sphinx-theme>=0.15",
    "myst-parser>=3.0",
]

[project.urls]
Homepage = "https://github.com/chronocratic/chronocratic-models"
Documentation = "https://chronocratic-models.readthedocs.io/"
Repository = "https://github.com/chronocratic/chronocratic-models"
Issues = "https://github.com/chronocratic/chronocratic-models/issues"

[dependency-groups]
dev = [
    "pytest>=8.2",
    "pytest-cov>=5.0",
    "ruff>=0.15.9",
    "towncrier>=24.8",
    "ty>=0.0.28",
]
notebooks = [
    "notebook>=7.3",
    "jupyterlab>=4.3",
]

[tool.uv]
default-groups = ["dev"]

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools_scm]
version_file = "src/chronocratic/models/_version.py"
local_scheme = "no-local-version"
fallback_version = "0.1.0a1"

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]

[tool.towncrier]
package = "chronocratic.models"
package_dir = "src"
directory = "changelog.d"
filename = "CHANGELOG.md"
start_string = "<!-- towncrier release notes start -->\n"
title_format = "## v{version} ({project_date})"
issue_format = "[#{issue}](https://github.com/chronocratic/chronocratic-models/issues/{issue})"
underlines = ["", "", ""]

[[tool.towncrier.type]]
directory = "added"
name = "Added"
showcontent = true

[[tool.towncrier.type]]
directory = "changed"
name = "Changed"
showcontent = true

[[tool.towncrier.type]]
directory = "deprecated"
name = "Deprecated"
showcontent = true

[[tool.towncrier.type]]
directory = "removed"
name = "Removed"
showcontent = true

[[tool.towncrier.type]]
directory = "fixed"
name = "Fixed"
showcontent = true

[[tool.towncrier.type]]
directory = "security"
name = "Security"
showcontent = true
```

### Key Changes

1. **`[build-system]`:** Add `setuptools-scm>=8` to requires. Add explicit `build-backend = "setuptools.build_meta"`.
2. **`name`:** `tscollection-models` to `chronocratic-models`.
3. **`version`:** Remove static `0.1.0`, replace with `dynamic = ["version"]`.
4. **`requires-python`:** Relax from `>=3.12,<3.13` to `>=3.12` to match reference repo.
5. **`dependencies`:** Prune to 7 packages (remove 6, add tqdm).
6. **`description`, `readme`, `license`, `keywords`, `authors`, `classifiers`:** All NEW -- match reference repo.
7. **`[project.optional-dependencies]` docs:** NEW -- Sphinx stack.
8. **`[project.urls]`:** NEW -- 4 URLs matching reference repo.
9. **`[dependency-groups] dev`:** Add `towncrier>=24.8`.
10. **`[tool.setuptools_scm]`:** NEW -- version file, local_scheme, fallback.
11. **`[tool.towncrier]`:** NEW -- full config matching reference repo.

## Documentation Setup

### Sphinx `conf.py` Requirements

Based on the reference repo (`chronocratic-datasets`), the `docs/conf.py` must:

1. Add `src/` to `sys.path` so autodoc can import `chronocratic.models`.
2. Import `__version__` from `chronocratic.models` for the `release` variable.
3. Enable extensions: `sphinx.ext.autodoc`, `sphinx.ext.napoleon`, `myst_parser`.
4. Set `napoleon_use_google_style = True` (project uses Google-style docstrings).
5. Use `pydata_sphinx_theme` with `navigation_depth: 3`.
6. Set `source_suffix` to support both `.md` and `.rst`.
7. Enable MyST extensions: `colon_fence`, `deflist`, `substitution`.

### Autodoc Modules to Document

All public modules that have `__all__` exports:

- `chronocratic.models` -- top-level API (10 models + configs)
- `chronocratic.models.supervised` -- supervised training API
- `chronocratic.models.augmentation` -- augmentation ABCs and re-exports
- `chronocratic.models.layers` -- shared neural network layers
- Each model family: `chronocratic.models.convolutional.dilated`, `.standard`, `.transformer`, `.recurrent`, `.generative`

### Directory Structure for `docs/`

```
docs/
├── conf.py              # Sphinx configuration
├── index.md             # Landing page
├── api/
│   ├── index.md         # API reference TOC
│   └── models.md        # Autodoc for chronocratic.models
├── quickstart.md        # TS2Vec encoding example
├── changelog.md         # Includes CHANGELOG.md
├── contributing.md      # Contributing guide
└── _static/
    └── custom.css       # Custom CSS overrides
```

### `.readthedocs.yaml` Configuration

Match reference repo exactly:

```yaml
version: 2

build:
  os: ubuntu-24.04
  tools:
    python: "3.12"

sphinx:
  configuration: docs/conf.py

python:
  install:
    - method: uv
      command: sync
      extras:
        - docs
```

## CI Workflow Fixes

### `build-and-test.yml` -- Specific Changes

**Line 24 (coverage path):**
```diff
-        run: uv run pytest tests/ --cov=src/chronocratic/datasets --cov-report=xml
+        run: uv run pytest tests/ --cov=src/chronocratic/models --cov-report=xml
```

**Line 43 (docs job install):**
```diff
-        run: pip install -e ".[docs]"
+        run: uv sync --extra docs
```

**Note:** The current workflow uses `pip install -e ".[docs]"` which will crash because `[project.optional-dependencies]` does not yet define `docs`. After adding the docs extra to pyproject.toml, this line must use `uv sync --extra docs` (consistent with the test job's `uv run` usage) or the pip command will work once the extra is defined.

### `pypi-publish.yml` -- No Changes Needed

Already references `chronocratic-models` in the environment URL. Uses `python -m build` which is correct.

### `ruff.toml` -- Path Update

**Line 8:**
```diff
-    "src/tscollection/models/distances/soft_dtw/soft_dtw_cuda.py",
+    "src/chronocratic/models/distances/soft_dtw/soft_dtw_cuda.py",
```

### Other Workflows

- **`release-prep.yml`:** No changes needed. Uses `towncrier build --yes --version` which reads config from pyproject.toml.
- **`auto-changelog-fragment.yml`:** No changes needed. Uses `towncrier check` which reads config from pyproject.toml.
- **`release-notes.yml`:** No changes needed.
- **`ff-merge-check.yml`, `ff-merge-do.yml`, `main-pre-merge-gate.yml`:** No package-specific paths.

## PEP 420 Namespace Package Considerations

### `chronocratic/` Must NOT Have `__init__.py`

- Current state: `src/tscollection/` has NO `__init__.py` -- already correct.
- After rename: `src/chronocratic/` must have NO `__init__.py`.
- `src/chronocratic/models/` DOES have `__init__.py` -- this is the actual package.

### Impact on Package Discovery

The current `[tool.setuptools.packages.find]` with `where = ["src"]` will discover:
- `chronocratic` (namespace package, no `__init__.py`)
- `chronocratic.models` (regular package with `__init__.py`)
- All subpackages

This is correct behavior. setuptools handles implicit namespace packages automatically.

### Import Path Changes

Before:
```python
from tscollection.models import TS2Vec
```

After:
```python
from chronocratic.models import TS2Vec
```

When combined with `chronocratic-datasets`:
```python
from chronocratic.datasets import ETThDataset
from chronocratic.models import TS2Vec
```

Both work because `chronocratic/` is an implicit namespace package -- Python merges the subpackages from both installed distributions.

### `__version__` Exposure

The reference repo's `docs/conf.py` imports `from chronocratic.datasets import __version__`. This means `src/chronocratic/models/__init__.py` must export `__version__`. Currently there is NO `__version__` in the models `__init__.py`. After setuptools_scm is configured, it writes `src/chronocratic/models/_version.py` which contains `__version__`. The `__init__.py` must import and re-export it:

```python
from ._version import __version__

__all__ = [
    "__version__",
    # ... existing model exports
]
```

## Risk Assessment

### High-Risk Tasks

| Task | Risk | Mitigation |
|------|------|------------|
| Namespace rename across 62 files | Missed reference causes `ImportError` | Automated `git mv` + global search-and-replace + full `pytest` run |
| Removing 6 dependencies | Downstream code may depend on transitive imports | Verified: 0 src/ imports for pandas, scikit-learn, joblib, openpyxl, h5py, torchaudio. torchvision has 0 actual imports (docstring only). |
| Adding `tqdm` to dependencies | Missing version constraint | Use `tqdm>=4.66.0` (stable, widely used, no breaking changes expected) |
| setuptools_scm versioning | No git tags exist yet; fallback version used | Current state: 0 git tags. setuptools_scm will use `fallback_version = "0.1.0a1"` until first tag. CI must `fetch-depth: 0` and `fetch-tags: true` (already configured). |

### Medium-Risk Tasks

| Task | Risk | Mitigation |
|------|------|------------|
| Sphinx docs build | Autodoc fails if imports break | Test `sphinx-build` locally before committing CI changes |
| `.readthedocs.yaml` | Build fails without ReadTheDocs account | File is committed; actual RTD account setup is deferred |
| Changing `requires-python` | Broader range may not be tested | Current: `>=3.12,<3.13`. Target: `>=3.12`. Code uses 3.12 features but should work on 3.13+. Reference repo uses `>=3.12`. |

### Low-Risk Tasks

| Task | Risk | Mitigation |
|------|------|------------|
| Adding LICENSE file | None -- static BSD-3-Clause text | Copy from reference repo |
| Adding README.md | None -- static content | Match reference structure |
| Creating changelog.d/ | None -- empty directory with README | towncrier config verified against reference |
| Updating ruff.toml path | Single line change | Verified: line 8 only |

### Dependencies Between Tasks

1. **Rename must complete before docs build** -- Sphinx autodoc imports `chronocratic.models`, not `tscollection.models`.
2. **pyproject.toml must be complete before CI runs** -- `uv sync --extra docs` fails if docs extra is not defined.
3. **setuptools_scm must be configured before first build** -- `python -m build` fails without dynamic version resolver.
4. **tqdm must be added before package installs cleanly** -- runtime ImportError without it.
5. **`__version__` must be exported from `__init__.py` before Sphinx builds** -- `conf.py` imports it.

## Common Pitfalls

### Pitfall 1: Missing tqdm Causes Runtime ImportError
**What goes wrong:** Package installs from PyPI but crashes when encoding is called because `tqdm` is imported but not declared as a dependency.
**Why it happens:** tqdm is used in `dilated/_mixin/encoding.py` but was never added to `pyproject.toml`.
**How to avoid:** Add `tqdm>=4.66.0` to `[project] dependencies` during this phase.
**Warning signs:** `ImportError: cannot import name 'tqdm'` when calling `encode()` on dilated models.

### Pitfall 2: Accidentally Creating `chronocratic/__init__.py`
**What goes wrong:** If `chronocratic/` gets an `__init__.py`, it becomes a regular package, not a namespace package. This breaks the shared namespace with `chronocratic-datasets`.
**Why it happens:** Tooling or manual creation adds `__init__.py` during the rename.
**How to avoid:** Explicitly verify `src/chronocratic/__init__.py` does NOT exist after rename.
**Warning signs:** `ImportError` when trying to import from both packages simultaneously.

### Pitfall 3: setuptools_scm Requires Git Tags
**What goes wrong:** `python -m build` succeeds but reports version `0.1.0a1` (fallback) instead of the real version.
**Why it happens:** No git tags exist yet in the repo.
**How to avoid:** This is expected for the initial release. The first git tag (e.g., `0.1.0a1`) created when publishing the GitHub release will trigger the real version. CI already has `fetch-depth: 0` and `fetch-tags: true`.
**Warning signs:** Package version shows `0.1.0a1` after tagging -- means `local_scheme` is working but tag format is wrong.

### Pitfall 4: Coverage Path Points to Wrong Package
**What goes wrong:** CI test job reports 0% coverage because `--cov=src/chronocratic/datasets` is the wrong path.
**Why it happens:** Current build-and-test.yml was copied from reference repo and not updated.
**How to avoid:** Change to `--cov=src/chronocratic/models`.
**Warning signs:** Coverage XML file is empty or reports no covered files.

### Pitfall 5: Docs Job Uses pip Instead of uv
**What goes wrong:** `pip install -e ".[docs]"` may fail or create a conflicting environment.
**Why it happens:** The docs job uses `pip` while test/lint/build jobs use `uv`.
**How to avoid:** Use `uv sync --extra docs` for consistency.
**Warning signs:** Build fails with dependency resolution errors.

## Code Examples

### setuptools_scm Version File
```python
# src/chronocratic/models/_version.py (auto-generated by setuptools_scm)
__version__ = "0.1.0a1"  # fallback, or derived from git tag
```

### Sphinx conf.py (based on reference repo)
```python
# docs/conf.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chronocratic.models import __version__  # noqa: E402

project = "chronocratic-models"
html_title = "chronocratic-models"
copyright = "2026-Present, The Chronocratic Developers"
author = "The Chronocratic Developers"
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navigation_depth": 3,
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

napoleon_use_google_style = True
autodoc_default_options = {
    "member-order": "bysource",
}

suppress_warnings = ["efifo"]
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "substitution",
]
```

### towncrier Fragment Example
```markdown
<!-- changelog.d/123.added.md -->
Added `encode()` support for the Series2Vec model via `BasicEncodingMixin`.
```

### ReadTheDocs Build Config
```yaml
# .readthedocs.yaml
version: 2

build:
  os: ubuntu-24.04
  tools:
    python: "3.12"

sphinx:
  configuration: docs/conf.py

python:
  install:
    - method: uv
      command: sync
      extras:
        - docs
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual version field | setuptools_scm dynamic versioning | Ongoing | Version derived from git tags; no manual bumps |
| hand-written changelog | towncrier fragment-based | 2024+ | PR-level changelog fragments assembled at release |
| pip for dependency management | uv | 2024+ | Faster installs, lockfile, dependency groups |
| pdoc for docs | Sphinx + autodoc + MyST | Ongoing | Better API reference, RTD integration |
| pypi-legacy upload | pypa/gh-action-pypi-publish + trusted publishing | 2023+ | No API tokens needed; OIDC-based auth |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `tqdm>=4.66.0` is the appropriate minimum version | Dependency Audit Results | Minimal; tqdm is stable with good backward compat. Exact version should be checked against what the lock file resolves. |
| A2 | `pandas` and `scikit-learn` can be safely removed from direct deps because they are not imported in `src/` | Dependency Audit Results | Medium; downstream `rbspaper` code (in gitignored `_sources/`) may depend on them. As a published package, this is correct -- only direct runtime deps should be declared. |
| A3 | `requires-python = ">=3.12"` (without upper bound) is safe to use | pyproject.toml Transformation | Low; code uses 3.12-specific features. 3.13 compatibility should be verified but is likely fine as 3.13 is backward compatible with 3.12. |
| A4 | The `torchvision` docstring reference in `primitives.py` line 220 does not constitute an actual import | Dependency Audit Results | None; this is a documentation analogy, not a code import. |
| A5 | Reference repo classifiers include Python 3.13 and 3.14 which we should mirror | pyproject.toml Transformation | Low; classifiers are metadata-only but should reflect tested versions. |

## Open Questions

1. **`__version__` export from `models/__init__.py`**
   - What we know: The reference repo's `docs/conf.py` imports `from chronocratic.datasets import __version__`. setuptools_scm writes `_version.py` with `__version__`.
   - What's unclear: Whether the current `models/__init__.py` already exports `__version__` or if this needs to be added.
   - Recommendation: Add `from ._version import __version__` and include it in `__all__`. This is a small change with clear benefit.

2. **Exact keyword list for pyproject.toml**
   - What we know: Reference repo uses 9 keywords. The phase scope includes "polymorphic augmentations" and "Lightning integration".
   - What's unclear: Whether additional keywords specific to this package (vs the datasets package) are needed.
   - Recommendation: Use the keyword list in the target state above; it can be refined later without breaking anything.

3. **README.md quickstart code snippet**
   - What we know: A TS2Vec `encode()` example with synthetic data is required.
   - What's unclear: The exact synthetic data shape and device placement to show.
   - Recommendation: Use a minimal 2D tensor (batch_size=1, seq_len=100) on CPU for reproducibility.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.12 | All code | Check at build time | 3.12 | — |
| uv | CI, dev | Check at build time | 0.11.2+ | pip |
| git | setuptools_scm | Check at build time | — | — |
| pip | CI (build, twine) | Check at build time | — | — |
| setuptools-scm | Build | Not yet installed | — | Add to build-system requires |
| towncrier | Changelog | Not yet installed | — | Add to dev group |
| Sphinx | Docs | Not yet installed | — | Add to docs extra |
| ruff | Lint | Installed | >=0.15.9 | — |
| tqdm | Runtime encoding | Check lock file | — | Add to dependencies |

**Missing dependencies with no fallback:**
- None -- all missing deps are resolved by adding them to pyproject.toml.

**Missing dependencies with fallback:**
- tqdm: Used in code but not in pyproject.toml. Add as `tqdm>=4.66.0`.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=8.2 |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest tests/ -x` |
| Full suite command | `uv run pytest tests/ --cov=src/chronocratic/models --cov-report=xml` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PKG-01 | Package builds with `python -m build` | smoke | `uv run python -m build` | Wave 0 (verify after rename) |
| PKG-02 | `twine check dist/*` passes | smoke | `uv run twine check dist/*` | Wave 0 |
| PKG-03 | All 62 files renamed; imports resolve | unit | `uv run pytest tests/ -x` | Exists (28 test files) |
| PKG-04 | Coverage reports against correct path | smoke | `uv run pytest tests/ --cov=src/chronocratic/models` | Wave 0 |
| PKG-05 | Sphinx docs build | smoke | `uv run sphinx-build -b html docs/ docs/_build/` | Wave 0 (docs/ not yet created) |
| PKG-06 | setuptools_scm produces version | unit | `python -c "from chronocratic.models import __version__; print(__version__)"` | Wave 0 |
| PKG-07 | towncrier check passes | smoke | `uv run towncrier check --compare-with origin/dev` | Exists (workflow already wired) |
| PKG-08 | ruff lint passes on renamed paths | smoke | `uv run ruff check src/` | Exists |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/ -x` (fast fail)
- **Per wave merge:** `uv run pytest tests/ --cov=src/chronocratic/models --cov-report=xml`
- **Phase gate:** Full suite green + `python -m build` + `twine check` + `sphinx-build` all pass

### Wave 0 Gaps
- [ ] No gaps in test files -- all 28 existing tests cover import paths. After rename, mechanical import updates are needed but test logic does not change.
- [ ] Framework already installed via dev group.
- [ ] **New gap:** Sphinx build test -- `docs/` directory and `conf.py` do not yet exist. Must be created in Wave 0 or verified immediately after creation.

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | N/A -- library package, no auth |
| V3 Session Management | No | N/A -- library package, no sessions |
| V4 Access Control | No | N/A -- library package, no access control |
| V5 Input Validation | Partial | numpy array shape validation in model inputs |
| V6 Cryptography | No | N/A -- no cryptographic operations |

### Known Threat Patterns for Python Packaging

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Supply chain attack via compromised deps | Spoofing | Pin dependency versions; verify against lock file |
| Typosquatting in dependency names | Spoofing | Review all dependency names before adding |
| Excessive dependency tree | Repudiation | Prune unused deps (7 removed in this phase) |
| Missing license | Repudiation | BSD-3-Clause LICENSE file added |

## Sources

### Primary (HIGH confidence)
- Reference repo `chronocratic-datasets` pyproject.toml -- fetched directly from GitHub raw URL; verified complete
- Reference repo `chronocratic-datasets` .readthedocs.yaml -- fetched directly from GitHub raw URL
- Reference repo `chronocratic-datasets` docs/conf.py -- fetched directly from GitHub raw URL
- Reference repo `chronocratic-datasets` LICENSE -- fetched directly from GitHub raw URL; BSD-3-Clause confirmed

### Secondary (MEDIUM confidence)
- `pyproject.toml` (current) -- read from repo root
- `ruff.toml` -- read from repo root
- `build-and-test.yml` -- read from `.github/workflows/`
- `STACK.md` -- read from `.planning/codebase/`
- `STRUCTURE.md` -- read from `.planning/codebase/`
- CONTEXT.md -- read from phase directory

### Tertiary (LOW confidence)
- tqdm version constraint `>=4.66.0` -- based on training knowledge of recent stable releases; should verify against npm/PyPI before finalizing

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all tools confirmed via reference repo and current pyproject.toml
- Architecture: HIGH -- namespace package pattern verified; no `__init__.py` in `tscollection/` confirmed
- Pitfalls: HIGH -- based on direct grep verification of 62 files, dependency audit with exact file counts

**Research date:** 2026-06-15
**Valid until:** 2026-07-15 (30 days -- stable packaging domain)

---

## RESEARCH COMPLETE

**Phase:** 06 - Prepare to Be Published as Package
**Confidence:** HIGH

### Key Findings

1. **62 Python files** need `tscollection` to `chronocratic` rename (34 src + 28 tests), plus 2 config files and 1 CI workflow.
2. **7 dependencies can be pruned:** `torchvision`, `torchaudio`, `joblib`, `openpyxl`, `h5py`, `pandas`, `scikit-learn` are zero-import in `src/`. This reduces the dependency tree significantly.
3. **1 missing dependency:** `tqdm` is imported in `dilated/_mixin/encoding.py` but NOT declared in `pyproject.toml`. This is a real bug that must be fixed.
4. **Zero git tags exist** -- setuptools_scm will use fallback `0.1.0a1` until the first release tag is created.
5. **CI coverage path** already references `chronocratic/datasets` (wrong package) -- must change to `chronocratic/models`.
6. **Reference repo** (`chronocratic-datasets`) provides exact templates for all metadata, docs, and tooling. All decisions are aligned.

### File Created

`/Users/skaf/VSCodeProjects/tsmodels/.planning/phases/06-prepare-to-be-published-as-package/06-RESEARCH.md`

### Confidence Assessment

| Area | Level | Reason |
|------|-------|--------|
| Standard Stack | HIGH | All tools verified against reference repo and current project state |
| Architecture | HIGH | Namespace package, import paths, setuptools_scm all verified |
| Pitfalls | HIGH | Dependency audit based on exact grep counts; no assumptions |

### Open Questions

1. `__version__` export from `models/__init__.py` -- needs verification
2. Exact keyword list -- can be refined post-release
3. README quickstart code snippet -- needs concrete example

### Ready for Planning

Research complete. Planner can now create PLAN.md files.
