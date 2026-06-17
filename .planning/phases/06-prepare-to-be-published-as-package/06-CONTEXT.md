# Phase 06: Prepare to Be Published as Package - Context

**Gathered:** 2026-06-15
**Status:** Ready for planning

## Phase Boundary

Make the library shippable to PyPI as `chronocratic-models`. Rename namespace from `tscollection` to `chronocratic`, add complete package metadata (description, authors, license, URLs), create LICENSE, README, CHANGELOG infrastructure, set up Sphinx documentation with ReadTheDocs hosting, and fix CI workflows to target the renamed paths.

**In scope:** Namespace rename (`tscollection` → `chronocratic`), pyproject.toml metadata, LICENSE (BSD-3-Clause), README.md, changelog.d/ + towncrier config, docs/ (Sphinx + MyST + pydata-theme), .readthedocs.yaml, CI workflow fixes, setuptools_scm versioning, **dependency audit** (remove unused deps from pyproject.toml).

**Out of scope:** Adding new models, changing model behavior, test logic changes (mechanical import updates only), GitHub Pages (using ReadTheDocs), pdoc (using Sphinx).

## Implementation Decisions

### D-01: Namespace Rename — Full, Single Phase

Rename `tscollection` → `chronocratic` across all source files, test files, imports, and configuration. Package dir becomes `src/chronocratic/models/`. PyPI package name: `chronocratic-models`. Import path: `from chronocratic.models import TS2Vec, ...`.

**Mechanical sweep:** Every `tscollection` reference → `chronocratic`. Includes:
- `src/tscollection/` → `src/chronocratic/`
- All import statements in src/ and tests/
- `pyproject.toml` package discovery paths
- CI workflow coverage paths
- `ruff.toml` source paths if any

### D-02: Namespace Package — Implicit PEP 420

`chronocratic/` will have **no `__init__.py`**. Both `chronocratic-datasets` and `chronocratic-models` contribute to the same namespace implicitly. Users install both packages and import from `chronocratic.datasets` and `chronocratic.models` independently.

### D-03: Version Strategy — setuptools_scm

Dynamic versioning from git tags. `[build-system]` adds `setuptools-scm>=8`. `version` becomes `dynamic = ["version"]`. `[tool.setuptools_scm]` writes `src/chronocratic/models/_version.py`. Matches reference repo (`chronocratic-datasets`).

### D-04: Documentation — Sphinx + pydata-theme + MyST + ReadTheDocs

Match `chronocratic-datasets` exactly:
- `docs/conf.py` — Sphinx config with autodoc, MyST parser
- `docs/index.md` — Landing page
- `docs/api/` — Auto-generated API reference
- `docs/quickstart.md` — TS2Vec encoding example
- `docs/changelog.md` — Includes CHANGELOG.md
- `docs/contributing.md` — Contributing guide
- `.readthedocs.yaml` — Build config (ubuntu-24.04, Python 3.12, uv)
- `[project.optional-dependencies]` — `docs = ["sphinx>=7.0", "pydata-sphinx-theme>=0.15", "myst-parser>=3.0"]`

### D-05: README — Match Reference Structure

- Badges: License (BSD-3-Clause), PyPI version, Python versions, downloads, build status, documentation, code style (ruff), GitHub stars
- Installation section with `pip install chronocratic-models`
- **Quick start:** TS2Vec encoding example (model instantiation + `encode()` call)
- **Models:** Catalog of all 10 models with one-line descriptions
- **Features:** Library highlights (polymorphic augmentations, Lightning integration, etc.)
- **Documentation:** Link to ReadTheDocs
- **License:** BSD 3-Clause

### D-06: LICENSE — BSD-3-Clause

Copy from reference repo. Copyright holder: "The Chronocratic Developers". Year: 2026-Present.

### D-07: Changelog — towncrier

Match reference repo config:
- `[tool.towncrier]` in pyproject.toml
- `changelog.d/` directory with type subdirs (added, changed, deprecated, removed, fixed, security)
- `changelog.d/README.md` — fragment instructions
- `CHANGELOG.md` — assembled by towncrier
- Existing `auto-changelog-fragment.yml` and `release-prep.yml` workflows already wired

### D-08: CI Workflow Fixes

- **build-and-test.yml:**
  - Coverage path: `src/chronocratic/datasets` → `src/chronocratic/models`
  - Docs job: Install `[docs]` extras (now defined), run `sphinx-build`
- **pypi-publish.yml:** Already correct (triggers on release, uses `python -m build`)
- **release-prep.yml:** Already correct (towncrier on dev branch)
- **auto-changelog-fragment.yml:** Already correct

### D-09: pyproject.toml — Complete Metadata

```toml
[project]
name = "chronocratic-models"
dynamic = ["version"]
description = "Self-supervised time-series representation learning models"
readme = "README.md"
requires-python = ">=3.12"
license = "BSD-3-Clause"
license-files = ["LICENSE"]
keywords = [...]
authors = [{name = "The Chronocratic Developers", email = "github@users.noreply.github.com"}]
classifiers = [...]
dependencies = [...]

[project.urls]
Homepage = "https://github.com/chronocratic/chronocratic-models"
Documentation = "https://chronocratic-models.readthedocs.io/"
Repository = "https://github.com/chronocratic/chronocratic-models"
Issues = "https://github.com/chronocratic/chronocratic-models/issues"
```

### D-10: Dependency Audit — Remove Unused Dependencies

Audit all `[project.dependencies]` and `[dependency-groups]` entries against actual imports in `src/`. Current dependencies:

| Dependency | Used in source? | Keep? |
|------------|----------------|-------|
| `numpy` | Yes — widely used | Keep |
| `pandas` | Check imports | TBD |
| `scipy` | Check imports | TBD |
| `scikit-learn` | Check imports | TBD |
| `lightning` | Yes — all models inherit LightningModule | Keep |
| `torch` | Yes — core dependency | Keep |
| `torchvision` | STACK.md says "installed but not directly referenced" | **Remove** |
| `torchaudio` | STACK.md says "installed but not directly referenced" | **Remove** |
| `joblib` | Check imports | TBD |
| `openpyxl` | Check imports | TBD |
| `h5py` | Check imports | TBD |
| `einops` | Yes — encoders use rearrange/reduce | Keep |
| `numba` | Check imports | TBD |

Planner should grep each dependency across `src/` to confirm usage. If a dep is only used in `tests/`, move it to `[dependency-groups] dev`. If unused entirely, drop it.

**Note:** `torchvision` and `torchaudio` are already flagged as unreferenced in STACK.md — high-confidence removals.

### Claude's Discretion

- Exact task wave ordering within the rename (src → tests → config → CI)
- README badge selection (match reference repo defaults)
- Model catalog ordering in README (by family: convolutional → transformer → recurrent → generative)
- Docs quickstart code snippet (exact TS2Vec example)
- Sphinx autodoc membership listing (which modules to document explicitly)
- `ruff.toml` source path updates (if needed after rename)

## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Reference Repository (Primary Template)
- **https://github.com/chronocratic/chronocratic-datasets** — Source of truth for README structure, pyproject.toml metadata, docs setup, LICENSE, CI patterns
- **https://raw.githubusercontent.com/chronocratic/chronocratic-datasets/main/pyproject.toml** — Full pyproject.toml including setuptools_scm, towncrier, docs deps
- **https://raw.githubusercontent.com/chronocratic/chronocratic-datasets/main/README.md** — README structure: badges, install, quick start, catalog, features
- **https://raw.githubusercontent.com/chronocratic/chronocratic-datasets/main/.readthedocs.yaml** — ReadTheDocs build config
- **https://raw.githubusercontent.com/chronocratic/chronocratic-datasets/main/LICENSE** — BSD-3-Clause text

### Project Context
- `.planning/PROJECT.md` — Core value: subclass-to-add, library-only scope
- `.planning/ROADMAP.md` — Phase 6: prepare to be published as package
- `.planning/REQUIREMENTS.md` — v1 requirements (AUG-01 through VER-05)
- `.planning/codebase/STACK.md` — Python 3.12, uv, setuptools, ruff, ty
- `.planning/codebase/STRUCTURE.md` — Directory layout, entry points, naming conventions
- `.planning/codebase/CONVENTIONS.md` — Import patterns, type annotations, docstrings

### Prior Phase Decisions
- `.planning/phases/01-augmentation-producer-contract/01-CONTEXT.md` — Augmentation contract, file layout, import patterns

## Existing Code Insights

### Reusable Assets
- `src/tscollection/models/__init__.py` — Public API barrel with `__all__` exports (10 models + configs). Rename to `src/chronocratic/models/__init__.py`.
- Existing GitHub workflows — `pypi-publish.yml`, `build-and-test.yml`, `release-prep.yml`, `auto-changelog-fragment.yml` already structured for PyPI publishing. Need path fixes, not rewrites.
- `ruff.toml` — Linter config. Update `line-length`, `target-version`, source paths if needed.
- All model classes have Google-style docstrings — autodoc will work out of the box.

### Established Patterns
- `kw_only=True` dataclasses for configs
- `__all__` exports in every module
- `TYPE_CHECKING` blocks to avoid circular imports
- Lazy imports in `__init__` methods to break circular deps
- Barrel `__init__.py` files at each package level

### Integration Points
- `src/` layout → rename `tscollection/` to `chronocratic/` (implicit PEP 420 namespace)
- `pyproject.toml` → add dynamic version, metadata, towncrier, setuptools_scm, docs deps
- CI workflows → update coverage paths, doc build commands
- Tests → mechanical import rename (`tscollection.models` → `chronocratic.models`)
- `graphify-out/` → regenerate after rename

## Specific Ideas

- Reference README shows namespace note: "The PyPI package name uses a hyphen (`chronocratic-models`), but the import uses the `chronocratic.models` namespace." Include this callout.
- Model catalog in README: group by family (convolutional/dilated, convolutional/standard, transformer, recurrent, generative) with one-line descriptions matching the model docstrings.
- Quick start: show TS2Vec `encode()` with synthetic data, matching the library-only scope (no runner, no DataModule).
- `towncrier` fragment types match reference: added, changed, deprecated, removed, fixed, security.

## Deferred Ideas

- **ReadTheDocs account setup** — Requires external account creation and repo linking. Do after package is built locally.
- **PyPI organization account** — `chronocratic` PyPI user/org. Set up before first publish, not in this phase.
- **Distribution to conda-forge** — Separate packaging concern. Future phase.
- **Pre-built wheels (manylinux)** — `cibuildwheel` integration. Future optimization.
- **Package size optimization** — Exclude large test fixtures, distances/ CUDA code. Future.

---

*Phase: 06-Prepare to Be Published as Package*
*Context gathered: 2026-06-15*
