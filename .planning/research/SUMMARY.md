# Research Summary

**Date:** 2026-05-21

## Stack

Follow timm + transformers patterns: config dataclasses, separate augmentation module, mixin for shared behavior. Avoid registry/factory patterns — library users prefer direct imports. Config-driven reproducibility without runner coupling.

## Features

Table stakes: clear import paths, extensible augmentations via subclassing, unified model API, full type hints. Core differentiator: strategy-based training where RIP/Adversarial are first-class objects, not hidden in model code. No runners, no CLI, no evaluation pipelines.

## Architecture

Augmentations are the product; models are the clients. `augmentation/` module has zero imports from `models/`. Split mixin hierarchy: `BaseEncodingMixin` → `PoolingEncodingMixin`, `DecompositionEncodingMixin`. Build order: foundation (mixin + config) → augmentation refactor → cleanup.

## Pitfalls

Critical risks:
1. **Behavior regression during refactor** — write smoke tests per model, run after each change
2. **Mixin diamond inheritance** — no `__init__` in mixins, verify MRO after changes
3. **Strategy coupling to model internals** — strategies are purely functional, tensor in → loss out
4. **Dead code after enum/factory removal** — final grep sweep + type check
5. **Config verbosity** — dataclasses are convenience, not mandatory. Models accept kwargs

## Phase Implications

Research confirms 3-phase structure:
- **Phase 1** — Foundation: split mixin, add config dataclasses (low risk, enables everything)
- **Phase 2** — Augmentation refactor: new ABCs, extract strategies, remove enums/factories (highest risk, needs smoke tests)
- **Phase 3** — Cleanup: remove rbspaper, polish API, verify imports (trivial, safety net)
