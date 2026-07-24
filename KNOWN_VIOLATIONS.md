# Known Violations

Internal tracking of code that does not yet match CONTRIBUTING.md rules. These are not in the public docs — fix them as you touch the relevant files.

Last audited: 2026-07-23

## Keyword-Only Signature Violations

Multi-param functions that don't use `*, ` (see CONTRIBUTING.md "Keyword-Only Signatures"):

### `autotcl/losses.py`
- `local_info_nce_loss(z1, z2, ...)` — loss function
- `info_nce_loss(z1, z2, ...)` — loss function
- `l1_out_loss(z1, z2, ...)` — loss function
- `maximum_mean_discrepancy_with_gaussian_kernel_loss(source, target, ...)` — loss function

### `chronocratic/models/normalization.py`
- `_normalize(...)` — internal helper
- `_denormalize(...)` — internal helper

### `augmentation/primitives.py`
- Various augmentation method signatures

### `tests/unit/test_encoding_output_shape.py`
- Multiple test functions use positional args (`model.encode(..., batch_size=2, num_workers=0)`)

## Config `kw_only` Violations

Not all config dataclasses use `@dataclass(kw_only=True)`. Check before relying on kw-only behavior.

## `pin_memory` Implementation Gaps

Contributing.md rule 5 says `pin_memory=True` only when no gradients flow. Current state:

- `BaseEncodingMixin` (dilated models): uses `pin_memory=data.device.type == 'cpu'` — guards GPU-resident data but doesn't check `gradient_enabled`
- `BasicEncodingMixin.encode()`: sets `pin_memory=True` unconditionally — violates the rule

## Device-Check Script Limitations

`scripts/check_device.sh` has a known false negative:
- Filters lines containing `.to(` but misses `torch.eye(...).to(device)` when constructor and `.to()` are on same line
- Check `autotcl/losses.py:199` as an example of a missed violation

## `save_hyperparameters` Ignore List

Contributing.md says `save_hyperparameters(ignore=["augmentation"])` is the standard. Some models may have additional ignored params — verify per model.

## Next Steps

- Fix keyword-only violations as you touch the files (don't batch-fix)
- Open issues for `pin_memory` gaps if they cause real problems
- Add `scripts/check_device.sh` improvements to the pre-commit hook