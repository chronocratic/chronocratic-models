# Technology Stack

**Analysis Date:** 2026-06-17

## Language and Runtime

**Primary:**
- Python 3.12 — All implementation code under `src/chronocratic/models/`
  - `requires-python = ">=3.12"` in `pyproject.toml`
  - Classifiers also list Python 3.13 support
  - Uses Python 3.12 features: `X | Y` union type syntax, `typing.override`

**Package Manager:**
- `uv` — Environment management, dependency resolution, task execution
- Lockfile: `uv.lock` present and committed
- Dependency groups: `dev` (default), `notebooks`, `docs`

## Core Frameworks

**Deep Learning:**
- **PyTorch** `>=2.4,<3.0` — Core tensor computations, autograd, neural network modules
  - `torch.nn` for layer definitions across all model families
  - `torch.fft` for Fourier transforms (CoST seasonal decoding)
  - `torch.optim.swa_utils.AveragedModel` for exponential moving average encoders (AutoTCL)
  - `torch.autograd.Function` for custom backward passes (Soft DTW CUDA)
- **PyTorch Lightning** `>=2.5,<3.0` — Training orchestration
  - All model classes inherit from `pl.LightningModule`
  - Manual optimization mode used for multi-optimizer training (e.g., AutoTCL two-phase training)
  - `Trainer`, `training_step`, `validation_step`, `configure_optimizers` across all models

**Tensor Operations:**
- **Einops** `>=0.8.2` — Tensor rearrangement via `rearrange`, `reduce`, `repeat`
  - Used in encoder implementations (`convolutional/dilated/encoders/encoders.py`), utility modules, and augmentation code

**Scientific Computing:**
- **NumPy** `>=2.1,<3.0.0` — Array operations, seeding in augmentation decorators
- **SciPy** `>=1.13.0` — Signal processing (`scipy.signal.butter`, `scipy.signal.lfilter`) for FFT-based frequency augmentation

**Performance:**
- **Numba** `>=0.65.1` — JIT compilation for CPU and CUDA paths
  - `src/chronocratic/models/distances/soft_dtw/soft_dtw_cuda.py`: CUDA-accelerated Soft DTW distance computation via `@jit` and `@cuda` decorators

**Progress Tracking:**
- **tqdm** `>=4.66.0` — Progress bars in encoding mixins (`_mixin/encoding.py`, `convolutional/dilated/_mixin/encoding.py`)

## Build System

**Backend:** `setuptools` `>=68` with `setuptools.build_meta`

**Versioning:** `setuptools-scm` `>=8` — Dynamic versioning from git tags
- Version file: `src/chronocratic/models/_version.py` (gitignored, generated at build)
- Fallback version: `0.1.0a1`
- Local scheme: `no-local-version`

**Package layout:** src-layout (`src/chronocratic/models/`)

**Changelog:** Towncrier `>=24.8` — Fragment-based changelog assembly
- Fragments: `changelog.d/` (types: added, changed, deprecated, removed, fixed, security)
- Output: `CHANGELOG.md`

## Dependencies

### Required (runtime)
| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | `>=2.1,<3.0.0` | Array operations, augmentation primitives |
| `scipy` | `>=1.13.0` | Signal processing (Butterworth filters) |
| `lightning` | `>=2.5,<3.0` | Training loop, optimizer hooks, logging |
| `torch` | `>=2.4,<3.0` | Neural network definitions, tensor ops |
| `einops` | `>=0.8.2` | Tensor reshaping/reduction |
| `numba` | `>=0.65.1` | JIT-compiled distance computation (Soft DTW) |
| `tqdm` | `>=4.66.0` | Progress bars during encoding |

### Dev
| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | `>=8.2` | Test runner |
| `pytest-cov` | `>=5.0` | Coverage reporting |
| `ruff` | `>=0.15.9` | Linting and formatting |
| `towncrier` | `>=24.8` | Changelog management |
| `ty` | `>=0.0.28` | Static type checking |

### Docs
| Package | Version | Purpose |
|---------|---------|---------|
| `sphinx` | `>=7.0` | Documentation generation |
| `pydata-sphinx-theme` | `>=0.15` | HTML documentation theme |
| `myst-parser` | `>=3.0` | Markdown source support for Sphinx |

### Notebooks
| Package | Version | Purpose |
|---------|---------|---------|
| `notebook` | `>=7.3` | Jupyter Notebook server |
| `jupyterlab` | `>=4.3` | JupyterLab IDE |

## Infrastructure

### Linting and Formatting
- **Tool:** Ruff `>=0.15.9`
- **Config:** `ruff.toml`
- **Line length:** 100
- **Target version:** `py312`
- **Rules:** `ALL` selected with ignores: D100, D101, D107, COM812, INP001, PLR0913, Q000, RET504, D212
- **Docstring convention:** Google style
- **Format:** Double quotes, space indent, no magic trailing comma
- **Excluded files:** `dependencies/`, `experiments_output/`, `soft_dtw_cuda.py`

### Testing
- **Framework:** pytest `>=8.2`
- **Config:** `pyproject.toml` — `testpaths = ["tests"]`, `pythonpath = ["."]`
- **Structure:** Flat tests in `tests/`, unit tests in `tests/unit/`, integration in `tests/integration/`
- **Coverage:** `--cov=src/chronocratic/models --cov-report=xml`
- **Shared fixtures:** `tests/conftest.py` (`train_steps`, `random_data`, `finite_losses`)

### CI/CD (GitHub Actions)
- **build-and-test.yml** — PR to `main`/`dev`: test, lint, build, docs jobs
- **pypi-publish.yml** — On GitHub Release: publish to PyPI via trusted publishing
- **release-prep.yml** — Manual dispatch: assemble changelog, open PR to `dev`
- **ff-merge-check.yml** / **ff-merge-do.yml** — Fast-forward merge enforcement
- **auto-changelog-fragment.yml** / **release-notes.yml** — Changelog automation

### Documentation
- **Generator:** Sphinx with `pydata_sphinx_theme`
- **Hosting:** ReadTheDocs (`.readthedocs.yaml` — Ubuntu 24.04, Python 3.12, uv)
- **Extensions:** `sphinx.ext.autodoc`, `sphinx.ext.napoleon`, `myst_parser`

### Knowledge Graph
- **graphify** — AST-based code graph. 1072 nodes, 2066 edges, 54 communities. Output in `graphify-out/`.

## Platform Requirements

**Development:** Python 3.12+, `uv` package manager. CPU-only tests are sufficient.

**Production:** CPU or GPU. PyTorch auto-dispatches. CUDA available via Numba JIT for Soft DTW when GPU present.

**License:** BSD-3-Clause
