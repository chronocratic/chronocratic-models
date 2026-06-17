# Phase 05 Dependencies

## Internal (within phase 05)
- Wave 2 depends on Wave 1 (all refactors must land before executor adjustments)

## External (cross-phase)
- **Input dependency**: Phase 04 must be complete (barrel exports removed, optional augmentation landed) ✓
- **Output dependency**: Phase 06 consumes phase 05's refactored executor and cleaned model APIs
- **Verification gap**: VER-01 through VER-05 smoke tests deferred to phase 07 (VER-05 already has detailed spec in phase 07/PLAN.md tasks VER-05.1–VER-05.6)

## Phase 07 Verification Backlink
See `../phase-07/PLAN.md` sections:
- VER-01: Basic instantiation and shape correctness
- VER-02: Load pre-trained weights verification
- VER-03: Pipeline integration verification
- VER-04: Encoding correctness verification
- VER-05: Smoke test suite for all models (see tasks VER-05.1 through VER-05.6)
