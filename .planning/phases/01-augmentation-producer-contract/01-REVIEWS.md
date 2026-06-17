---
phase: 1
reviewers: [claude]
reviewed_at: 2026-06-12T11:01:52Z
plans_reviewed: [01-01-PLAN.md, 01-02-PLAN.md, 01-03-PLAN.md, 01-04-PLAN.md, 01-05-PLAN.md, 01-06-PLAN.md, 01-07-PLAN.md, 01-08-PLAN.md, 01-09-PLAN.md, 01-10-PLAN.md, 01-11-PLAN.md, 01-12-PLAN.md, 01-13-PLAN.md]
note: claude self-review forced via --claude despite running inside Claude Code (no other independent CLI available); independence is reduced.
---

# Cross-AI Plan Review — Phase 01

## Claude Review

# Cross-AI Plan Review — Phase 01 Augmentation Producer Contract

## 1. Summary

Plans are well-structured, faithful to SPEC, TDD-driven, with clean wave layout and good per-task `must_haves`/`key_links`. Design intent (N×M→N+M, typed ViewSets, branchless models, ISP trainable gate) is correctly decomposed across 13 plans. **But two HIGH ordering bugs will break imports mid-migration**, one **HIGH typing bug** in the TS2Vec wiring contradicts the SPEC and fails `ty`, and the **numerical-equivalence verification (SC-7/G6) does not actually compare against pre-refactor behavior** — it compares seeded runs to each other, which proves determinism, not preservation. These are fixable but must be addressed before execution.

## 2. Strengths

- Bottom-up sequencing (build → wire → delete) is the right shape; D-05 keep-old-until-final-delete is sound.
- Per-plan `must_haves.truths` + `key_links` give checkable gates; `min_lines` guards stop empty test stubs.
- TDD RED→GREEN explicit in every build plan.
- Import-hygiene enforced by an actual test (01-12 reads source, asserts no `convolutional` substring) — good mechanical check for G1.
- Scope discipline: deferred ideas (registry, HOF, plan/produce) explicitly excluded. No over-engineering.
- Trainable gate centralized in `trainable_support.py`; only place with `isinstance`. Matches D-02.

## 3. Concerns

**HIGH — Delete ordering inverted (plan 11 before plan 13).**
Plan 11 (wave 5) removes `AugmentationMethod`/`TrainingViews`/`TrainableAugmentation` from `base.py` and deletes `dual.py`. But `cost/augmentation.py` (05/06 keep import per D-05), `ts2vec/augmentation.py`, `autotcl/methods.py`, `autotcl/utils.py`, and `test_augmentation_base.py` still `from ...base import AugmentationMethod/TrainingViews` until plan 13 (wave 6). The barrel imports those per-model modules → loading it after plan 11 raises `ImportError`. Plan 11's own verify (`python -c "from tscollection.models.augmentation import ..."`) loads the barrel and **would itself fail**. Consumers must lose their imports *before* the definition is deleted. Fix: swap order (13 before 11) or merge into one atomic wave.

**HIGH — `temporal_unit` passed to `produce()` breaks the Protocol and contradicts SPEC.**
Plan 05 action: `pair = self._augmentation.produce(x, temporal_unit=self._temporal_unit)`. The contract is `AugmentationProducer.produce(self, x) -> V` (no `temporal_unit`); TS2Vec holds the field as `AugmentationProducer[AlignedPair]`. Calling with an extra kwarg fails `ty` (SC-6). SPEC §4.7 sketch is `produce(x)` with **no** temporal_unit — crop granularity belongs baked into `CropShiftProducer.__init__` via its params, not passed per-call. Fix: drop the kwarg; configure `temporal_unit` at producer construction.

**HIGH — SC-7 / G6 "numerical behavior unchanged" is not verified.**
Plans 05–08 each describe "Seeded model produces identical loss to unseeded run" (nonsensical) or "matches old" but the tests as written compare two new seeded runs to each other = determinism, not equivalence to the **pre-refactor** path. No plan captures a baseline from `main`/old code. G6 is the only correctness guarantee for behavior preservation and it has no real gate. Fix: capture seeded old-path outputs (git stash / baseline tensors committed as fixtures) and assert new == baseline, or explicitly downgrade the claim.

**MEDIUM — `TSTCCDualAugmentation` kept as a self-contradictory alias (08).**
Plan 08 step 3 "keep `TSTCCDualAugmentation = ...`" while step 5 "delete `DualAugmentation` import". A subclass can't survive without its base; the "alias" is underspecified and `dual.py` is deleted in 11 anyway. Keeping a broken symbol across two waves adds risk for ~zero benefit (tests migrated in 10, model rewired in 08). Fix: delete `TSTCCDualAugmentation` in 08, or define the alias as a no-op that needs no `DualAugmentation`.

**MEDIUM — `Seeded` determinism is torch-only; the two most important producers use numpy.**
`Seeded.produce` forks only `torch.random`. `CropShiftProducer` and `CosTRandomFunctionAugmentation` draw from `np.random`. Wrapping them in `Seeded` does **not** make them reproducible. Tests pass only because they wrap `Jitter` (torch). Flag the limitation, or fork numpy RNG too.

**MEDIUM — covariant `TypeVar` in parameter position (04).**
`V = TypeVar("V", covariant=True)` then `Seeded.__init__(self, *, inner: AugmentationProducer[V], ...)`. Covariant vars in non-constructor parameter position are rejected by type checkers; `__init__` is usually exempt but behavior under `ty` is unconfirmed. Verify `ty` accepts it; if not, use an invariant `TypeVar` for `Seeded`.

**MEDIUM — `_eval_mutual_information` adds two `isinstance` checks, in tension with SC-3/G3.**
SC-3: "No model body contains `isinstance(...augmentation...)` except AutoTCL's single trainable gate." Plan 07 adds `isinstance(TrainableAugmentationProducer)` at two spots in the diagnostic method with a §4.5.1 comment. That's three gates total, not one, and not via the helper. Reconcile SC-3 wording or route eval-mode toggling through a helper.

**LOW — 01-03 `key_links` claims `producers.py` imports `primitives.py`.**
Frontmatter `key_link` pattern `from.*primitives import` contradicts the action body ("imports only from base.py") and SPEC §4.9 (producers import nothing but base). Harmless but the checker may flag a missing/extra link. Remove the link.

**LOW — `FullOverlapPair` no-op slice assumes encoder preserves time length.**
`overlap_length = x.size(1)` (input T) used to slice the **embedding** time axis. Valid only if encoder time-length == input length (true for TS2Vec dilated conv). Note the assumption; a downsampling encoder would mis-slice.

## 4. Suggestions

- Reorder deletion: make per-model/import cleanup (current 13) run **before** core deletion (current 11), or fuse them into one wave with a single `grep -rn 'AugmentationMethod|TrainingViews|DualAugmentation' src/ tests/ == 0` gate plus full `uv run pytest` + `ty check src/`.
- Remove `temporal_unit` from `produce()`; add it to `CropShiftProducer(__init__)` and default `CropShiftProducer()` from `temporal_unit` the model passes at construction.
- Add a real equivalence fixture: before refactor, run each model 5 seeded steps on `main`, save losses/embeddings as `tests/fixtures/*_baseline.pt`, assert `torch.allclose` in 05–08. Otherwise relabel SC-7 as "determinism only."
- Make `Seeded` fork numpy RNG (`np.random.get_state`/`set_state`) or document torch-only and forbid wrapping crop/cost producers in a test.
- Collapse 08's `TSTCCDualAugmentation` decision: delete now (tests already migrated by 10) unless a concrete external importer exists.
- Add an explicit `ty check src/` (whole tree) gate at the end of waves 5 and 6 — current 11 only checks `augmentation/`, which is exactly why the cross-module import break would slip through.

## 5. Risk Assessment

**Overall: MEDIUM-HIGH.**

Design is sound and decomposition is good, so the ceiling is high. But as written, **plan 11 cannot pass its own verification** (import cascade), **plan 05 cannot pass `ty`** (temporal_unit), and the headline guarantee (G6 behavior preservation) is unverified. None require redesign — they're ordering, one signature, and one missing baseline fixture. Fix those four and risk drops to LOW. Ship-blocking until the two HIGH import/typing bugs and the SC-7 gap are resolved.

---

## Consensus Summary

Single reviewer (claude). Findings stand as-is; no cross-reviewer consensus available.

### Agreed Concerns (ship-blocking)

1. **HIGH — Delete ordering inverted (plan 11 before 13):** core symbols (`AugmentationMethod`/`TrainingViews`/`TrainableAugmentation`, `dual.py`) deleted in wave 5 while per-model modules + tests still import them until wave 6. Barrel import → `ImportError`; plan 11's own verify fails. Fix: 13 before 11, or fuse into one atomic wave.
2. **HIGH — `temporal_unit` kwarg breaks the Protocol (plan 05):** `produce(x, temporal_unit=...)` violates `AugmentationProducer.produce(self, x) -> V` and SPEC §4.7; fails `ty` (SC-6). Fix: bake `temporal_unit` into `CropShiftProducer.__init__`.
3. **HIGH — SC-7/G6 behavior-preservation unverified:** tests compare two new seeded runs (determinism), not new-vs-pre-refactor baseline. Fix: capture baseline tensors from old path as fixtures, assert `torch.allclose`; else relabel SC-7 "determinism only."

### Secondary Concerns

- MEDIUM — `TSTCCDualAugmentation` self-contradictory alias (08): kept while its base import is deleted.
- MEDIUM — `Seeded` determinism is torch-only; `CropShiftProducer`/`CosTRandomFunctionAugmentation` use `np.random` and stay non-reproducible.
- MEDIUM — covariant `TypeVar` in `Seeded.__init__` param position (04): confirm `ty` accepts it.
- MEDIUM — `_eval_mutual_information` adds 2 `isinstance` checks (07), in tension with SC-3 "single gate."
- LOW — 01-03 `key_link` claims `producers.py` imports `primitives.py`, contradicting action body + SPEC §4.9.
- LOW — `FullOverlapPair` no-op slice assumes encoder preserves time length.

### Recommended Action

Resolve the 3 HIGH items before execution (ordering swap, drop `temporal_unit` kwarg, add equivalence fixture or downgrade SC-7). Add whole-tree `ty check src/` gate at end of waves 5 and 6. Risk drops MEDIUM-HIGH → LOW once fixed.

### Divergent Views

None (single reviewer).
