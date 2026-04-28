#!/usr/bin/env bash
# Run game-side checks used by the LoRA / benchmark cohort (requires fallen-empire checkout).
set -euo pipefail
REPO="${SOURCE_REPO:-$HOME/fallen-empire}"
if [[ ! -d "$REPO" ]]; then
  echo "Set SOURCE_REPO to your fallen-empire game directory (missing: $REPO)" >&2
  exit 1
fi
cd "$REPO"
if [[ ! -f package.json ]]; then
  echo "Not a Node project: $REPO" >&2
  exit 1
fi
npm run test:ml-cohort
