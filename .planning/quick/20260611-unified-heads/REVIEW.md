---
phase: quick-20260611-unified-heads
reviewed: 2026-06-11T00:00:00Z
depth: deep
files_reviewed: 23
files_reviewed_list:
  - src/tscollection/models/_finetuning/__init__.py
  - src/tscollection/models/_finetuning/adapters.py
  - src/tscollection/models/_finetuning/callbacks.py
  - src/tscollection/models/_finetuning/factory.py
  - src/tscollection/models/_finetuning/finetuning.py
  - src/tscollection/models/_finetuning/utils.py
  - src/tscollection/models/convolutional/standard/series2vec/__init__.py
  - src/tscollection/models/convolutional/standard/series2vec/model.py
  - src/tscollection/models/convolutional/standard/series2vec/network.py
  - src/tscollection/models/convolutional/standard/tstcc/__init__.py
  - src/tscollection/models/convolutional/standard/tstcc/config.py
  - src/tscollection/models/convolutional/standard/tstcc/model.py
  - src/tscollection/models/generative/timevae/vae_base.py
  - src/tscollection/models/transformer/tst/__init__.py
  - src/tscollection/models/transformer/tst/model.py
  - tests/integration/__init__.py
  - tests/integration/test_finetuning_integration.py
  - tests/unit/test_backbone_representation_dim.py
  - tests/unit/test_finetuning_package.py
  - tests/unit/test_series2vec_finetuning.py
  - tests/unit/test_tst_finetuning.py
  - tests/unit/test_tstcc_finetuning.py
findings:
  critical: 1
  warning: 0
  info: 5
  total: 6
status: issues_found
---

# Re-review: Unified Downstream Heads Refactoring

**Reviewed:** 2026-06-11T00:00:00Z
**Depth:** deep (cross-file analysis, import graphs, call chains)
**Files Reviewed:** 23 (15 source + 8 test)
**Scope:** Last 14 commits on `feature/baseline`
**Status:** issues_found

## Summary

Re-reviewed after the previous round of fixes (CR-01, WR-01 through WR-06). All previously flagged warnings and the original critical (integration test out-of-bounds targets) have been correctly addressed. The factory signatures now use `RepresentationBackbone`, task validation is in place, Series2Vec uses the public `representation_dim` property, and integration tests use `idx % num_classes` for targets.

Found **1 critical crash** that was not fixed by the previous round: `classification_loss` still uses `.squeeze()` (no dim argument), which produces a 0-D scalar when `batch_size=1` and targets is `(1,)`. This is the same root issue as the previous WR-06, but it is a confirmed crash (BLOCKER), not a theoretical concern. Additionally found **5 info-level** issues.

## Critical Issues

### CR-01: `classification_loss` crashes with `batch_size=1` when targets are 1-D

**File:** `src/tscollection/models/_finetuning/utils.py:33`
**Severity:** BLOCKER

**Issue:** The function uses `targets.long().squeeze()` (no `dim` argument), which removes ALL dimensions of size 1. When `batch_size=1` and targets is `(1,)` (1-D tensor of a single element), `squeeze()` collapses it to a 0-D scalar. `nn.functional.cross_entropy` crashes:

```
ValueError: Expected input batch_size (1) to match target batch_size (0).
```

This occurs whenever the last batch of a DataLoader has only 1 sample — a common situation when dataset size is not divisible by batch_size, or when validation sets are smaller than the configured batch_size.

Empirical verification:
```python
>>> t = torch.tensor([1])  # shape (1,), batch_size=1
>>> t.squeeze()             # shape (), ndim=0 — 0-D scalar
tensor(1)
>>> nn.functional.cross_entropy(torch.tensor([[0.3, 0.7]]), t.squeeze().long())
ValueError: Expected input batch_size (1) to match target batch_size (0).
```

The docstring at lines 21-24 explicitly warns about this edge case ("avoid the 0-D scalar edge case at batch_size=1") and recommends `view(-1)`, but the implementation at line 33 still uses `squeeze()`.

This was previously flagged as WR-06 but was not fixed — the fix commit addressed the integration test's out-of-bounds targets (the old CR-01) but did not change the squeeze behavior in the source code.

**None of the current integration or unit tests exercise `batch_size=1`,** so the crash is not caught by the test suite.

**Fix:** Use `view(-1)` as the docstring recommends — it handles all target shapes without risk of dimension collapse:

```python
# Current (crashes at batch_size=1 with 1-D targets)
return nn.functional.cross_entropy(predictions, targets.long().squeeze())

# Fixed (safe for (B,), (B,1), and (B,1,1) target shapes)
return nn.functional.cross_entropy(predictions, targets.long().view(-1))
```

Alternatively, use `squeeze(-1)` for a more surgical fix that only removes the trailing dimension:

```python
return nn.functional.cross_entropy(predictions, targets.long().squeeze(-1))
```

---

## Warnings

(No warnings. Previous WR-01 through WR-06 have been addressed; the squeeze issue has been elevated to CR-01 above.)

---

## Info

### IN-01: `_get_optimizer` return type should be `type[torch.optim.Optimizer]`

**File:** `src/tscollection/models/convolutional/standard/series2vec/model.py:27`
**Severity:** Info

**Issue:** The function returns optimizer *classes* (`torch.optim.Adam`, `torch.optim.RAdam`, `torch.optim.AdamW`), not instances. The return type `Callable[..., torch.optim.Optimizer]` is technically correct but imprecise — `type[torch.optim.Optimizer]` more accurately conveys "returns a constructor, not an instance."

```python
# Current
def _get_optimizer(name: str) -> Callable[..., torch.optim.Optimizer]:

# Preferred
def _get_optimizer(name: str) -> type[torch.optim.Optimizer]:
```

### IN-02: `Series2VecNetwork.representation_dim` naming ambiguity

**File:** `src/tscollection/models/convolutional/standard/series2vec/network.py:128`
**Severity:** Info

**Issue:** `Series2VecNetwork` exposes `representation_dim` returning the per-branch dimension (e.g., 4), while `Series2Vec` (the LightningModule) exposes `representation_dim` returning `2 * self.network.representation_dim` (e.g., 8). Both satisfy the `RepresentationBackbone` protocol.

If someone accidentally passes a `Series2VecNetwork` instance (instead of `Series2Vec`) to a factory, the head would be sized for half the actual representation, causing a shape mismatch at forward time. The docstring does clarify this, but the naming overlap is fragile. Consider renaming the network-level property to `branch_representation_dim`.

### IN-03: `tstcc_representations` computes unused logits

**File:** `src/tscollection/models/_finetuning/adapters.py:95`
**Severity:** Info

**Issue:** `tstcc_representations` calls `backbone(x.float())`, which runs the full TCCEncoder forward pass — including 3 conv+pool blocks and the logits Linear layer — then discards logits: `_logits, features = backbone(x.float())`. The logits computation wastes FLOPs during fine-tuning.

This is a design constraint of the current TCCEncoder (no feature-only path exists). Addressing it requires backbone-level changes and is out of scope for this review. Noted for future optimization.

### IN-04: `regression_loss` docstring missing Google-style Args/Returns

**File:** `src/tscollection/models/_finetuning/utils.py:36-46`
**Severity:** Info

**Issue:** The `regression_loss` function (renamed from the previous `_mse_loss`) has a one-line summary ("MSE loss for regression tasks.") but includes no Args/Returns sections. The sibling `classification_loss` function at line 18 does include these. CLAUDE.md requires Google-style docstrings with structured parameter and return documentation.

```python
# Current
def regression_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MSE loss for regression tasks."""
    return nn.functional.mse_loss(predictions, targets)

# Fixed
def regression_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MSE loss for regression tasks.

    Args:
        predictions: Model outputs of shape ``(B, num_outputs)``.
        targets: Ground truth values of shape ``(B, num_outputs)``.

    Returns:
        Scalar MSE loss.
    """
    return nn.functional.mse_loss(predictions, targets)
```

### IN-05: Underscore-prefixed module import in integration tests

**File:** `tests/integration/test_finetuning_integration.py:12`
**Severity:** Info

**Issue:** `from tscollection.models import _finetuning` imports the underscore-prefixed (internal) module into the public integration test suite. The bare `_finetuning` reference is used in `TestBarrelExportsClean` to verify `__all__` exports match. This couples integration tests to a private API boundary; if the module is reorganized, tests break even if the public API is unchanged.

For testing purposes this is pragmatic, but worth noting that integration tests are validating internal package structure rather than user-facing behavior.

---

## Cross-File Analysis

### Import Graph

No circular dependencies. Clean one-way flow:
```
Backbone models (TST, Series2Vec, TSTCC)
    →  Reference _finetuning in docstrings ONLY (no code imports)

_finetuning package internals:
    adapters.py  ──→  torch (external)
    callbacks.py ──→  lightning.pytorch (external)
    finetuning.py ──→  lightning.pytorch, torch (external)
    factory.py ──→  adapters.py, finetuning.py, utils.py
    utils.py ──→  torch (external)
    __init__.py ──→  all submodules (re-exports)
```

Backbone models do NOT import `_finetuning` — they reference it only in docstrings. This maintains clean separation: the fine-tuning layer depends on backbones (via `RepresentationBackbone` protocol), not the reverse.

### Previously Flagged Issues — Resolution Status

| Finding | Status | Notes |
|---------|--------|-------|
| CR-01 (old): Integration test out-of-bounds targets | Fixed | `idx % self.num_classes` used in all 3 datasets |
| WR-01: Factory backbone type `FineTuningModule` | Fixed | Now `RepresentationBackbone` |
| WR-02: `num_classes` vs `num_outputs` inconsistency | Fixed | All factories use `num_outputs` |
| WR-03: Missing task validation | Fixed | `_validate_task()` shared helper |
| WR-04: Series2Vec `_representation_dims` direct access | Fixed | Uses `self.network.representation_dim` |
| WR-05: `assert True` in integration test | Fixed | Proper `callback_metrics` assertion |
| WR-06: `squeeze()` crash risk | NOT FIXED | Elevated to CR-01 above |

### Test Coverage Gaps

1. **No test exercises `batch_size=1`** — the crash in CR-01 is uncaught because none of the integration or unit tests use `batch_size=1`. Adding a test with `batch_size=1` or an odd dataset size (e.g., 3 samples with batch_size=4) would catch this.

2. **No test verifies head gradients when backbone is frozen** — `test_freeze_backbone_true_optimizer_sees_only_head_params` verifies the optimizer only contains head params, but does not verify head params receive non-None gradients after `loss.backward()`. The complementary unfrozen test (`test_freeze_backbone_false_backbone_receives_grads`) does check gradient flow.

3. **Factory unit tests do not exercise adapters** — `TestFactoryFunctions` in `test_finetuning_package.py` uses `_DummyBackbone` stubs that satisfy the protocol but do not test the model-specific adapters (batch unpacking, representation extraction). This is intentional (unit test isolation), and the real backbone smoke tests in `test_backbone_representation_dim.py` cover the end-to-end path.

---

_Reviewed: 2026-06-11T00:00:00Z_
_Reviewer: Claude (gsd-code-review)_
_Depth: deep_
