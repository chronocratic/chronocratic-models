# External Integrations

**Analysis Date:** 2026-06-17

## Third-Party Services

**None at runtime.** This is a self-contained library for time-series representation learning. No external APIs, databases, or cloud services are called during model execution.

## Library Integrations

### Deep Learning Stack
- **PyTorch** (`torch`) — All model classes are `torch.nn.Module` subclasses. Used for tensor operations, autograd, and custom `torch.autograd.Function` (Soft DTW).
- **PyTorch Lightning** (`lightning.pytorch`) — All models are `pl.LightningModule` subclasses. Provides `Trainer`, training/validation loops, optimizer configuration, and `self.log()` metric tracking. Manual optimization (`automatic_optimization = False`) used in AutoTCL for two-phase training.
- **Einops** (`einops`) — `rearrange`, `reduce`, `repeat` for flexible tensor layout transformations (NLC/NCL channel formats).

### Scientific Computing
- **NumPy** (`numpy`) — Array operations, random seeding in augmentation decorators (`augmentation/decorators.py`).
- **SciPy** (`scipy.signal`) — Butterworth filter design (`butter`) and filtering (`lfilter`) for frequency-domain augmentation primitives.
- **Numba** (`numba`, `numba.cuda`) — JIT-compiled CPU paths and CUDA kernels for Soft DTW distance computation (`distances/soft_dtw/soft_dtw_cuda.py`). MIT-licensed third-party code embedded verbatim.

### Progress and Logging
- **tqdm** — Progress bars during batch encoding in `_mixin/encoding.py` and `convolutional/dilated/_mixin/encoding.py`.
- **Python `logging`** — Standard library logger used in encoding mixins for batch progress.

## Data Formats

**Input:**
- `torch.Tensor` — Primary input format. Shapes: `(batch, channels, seq_len)` (NCL) or `(batch, seq_len, features)` (NLC).
- `numpy.ndarray` — Accepted and converted to tensors by augmentation primitives and encoding mixins.

**Output:**
- `torch.Tensor` — Model representations, latent vectors, reconstructed data.

**Serialization:**
- PyTorch `state_dict()` / `load_state_dict()` — Model weight persistence.
- Lightning checkpoint files (`.ckpt`) — Full training state (weights, optimizer, hyperparameters).

**No external file I/O** (HDF5, Excel, Joblib) is part of this library. Those belong to downstream consumer packages.

## Publication and Distribution

**Package registry:** PyPI (`chronocratic-models`)
- Homepage: `https://github.com/chronocratic/chronocratic-models`
- Documentation: `https://chronocratic-models.readthedocs.io/`

**Build pipeline:**
- `python -m build` produces source distribution and wheel
- `twine check` validates before publishing
- Trusted publishing via `pypa/gh-action-pypi-publish@release/v1` on GitHub Release events

**Versioning:** Semantic versioning via git tags (`setuptools-scm`)
- Changelog assembled by towncrier from `changelog.d/` fragments

## CI/CD Integrations

**Platform:** GitHub Actions (`.github/workflows/`)

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `build-and-test.yml` | PR to `main`/`dev` | Run tests, lint, build, docs |
| `pypi-publish.yml` | GitHub Release published | Publish package to PyPI |
| `release-prep.yml` | Manual dispatch | Assemble changelog, open PR |
| `ff-merge-check.yml` | Push/PR to `main` | Enforce fast-forward only merges |
| `ff-merge-do.yml` | Manual dispatch | Execute fast-forward merge |
| `auto-changelog-fragment.yml` | PR merge | Auto-generate changelog fragments |
| `release-notes.yml` | Various | Changelog automation |

**Secrets used:**
- `FF_MERGE_TOKEN` — Personal access token for PR creation (required because `GITHUB_TOKEN` cannot create/approve its own PRs)

**ReadTheDocs Integration:**
- `.readthedocs.yaml` configures automated documentation builds
- Ubuntu 24.04, Python 3.12, `uv sync --extra docs`

## Authentication & Identity

**Not applicable.** No user authentication, API keys, or identity management.

## Environment Configuration

**Required env vars:** None. The library does not read environment variables at runtime.

**Secrets:** No secrets files. The `.env` pattern is gitignored but no `.env` file exists.

## Webhooks & Callbacks

**Incoming:** None
**Outgoing:** None

## External Code Attribution

The library incorporates third-party code under permissive licenses:

- **Soft DTW CUDA** — `src/chronocratic/models/distances/soft_dtw/soft_dtw_cuda.py` — MIT License (Mehran Maghoumi, 2020). Embedded verbatim; excluded from ruff linting.
- **TS2Vec** — Based on `zhihanyue/ts2vec`
- **CoST** — Based on `salesforce/CoST`
- **AutoTCL** — Based on `AslanDing/AutoTCL`

## Dependency Risk Summary

| Package | Risk Level | Notes |
|---------|------------|-------|
| `torch` | Low | Stable major version pin (`>=2.4,<3.0`) |
| `lightning` | Low | Stable major version pin (`>=2.5,<3.0`) |
| `numba` | Medium | CUDA ABI compatibility can break across GPU driver versions |
| `numpy` | Low | Major version pin (`>=2.1,<3.0.0`) |
