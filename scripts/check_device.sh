#!/usr/bin/env bash
# check_device.sh — Advisory lint guard for bare tensor constructors
#
# Detects torch.eye/zeros/ones/arange/full/tensor calls without device=
# in model source files. Lines annotated with "# device-ok" are suppressed.
#
# Exit 0 = clean, Exit 1 = findings.
set -euo pipefail

SRC_DIR="${1:-src/chronocratic/models}"

if [ ! -d "$SRC_DIR" ]; then
  echo "ERROR: source directory not found: $SRC_DIR" >&2
  exit 1
fi

# Match bare torch constructors (not *_like) without device= on the same line
HITS=$(grep -rn \
  --include='*.py' \
  -E 'torch\.(eye|zeros|ones|arange|full|tensor|randn|rand|empty|randsign|randint|randperm)\(' \
  "$SRC_DIR" \
  2>/dev/null \
  | grep -v '_like(' \
  | grep -v 'device=' \
  | grep -v '# device-ok' \
  | grep -v '# noqa:' \
  | grep -v '\.to(' \
  | grep -v 'register_buffer' \
  | grep -v 'nn\.Parameter' \
  | grep -v '^\s*#' \
  | grep -v '>>>' \
  | grep -v '__pycache__' \
  || true)

if [ -z "$HITS" ]; then
  echo "PASS: no bare tensor constructors found in $SRC_DIR"
  exit 0
fi

echo "FAIL: bare tensor constructors without device= found:"
echo "$HITS"
echo ""
echo "Fix: add device= kwarg or annotate with '# device-ok' if intentional."
exit 1
