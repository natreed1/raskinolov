#!/usr/bin/env bash
# Build JSONL splits then run mlx_lm.lora. Extra args go to mlx_lm.lora only (override YAML).
# Prefer: python scripts/ml_workflow.py train …  (writes per-run docs + docs/run_history.md)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ ! -d .venv ]]; then
  echo "Create .venv first: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi
# shellcheck source=/dev/null
source .venv/bin/activate

SYNTH=()
LORA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --synthetic-smoke)
      SYNTH=(--synthetic-smoke)
      shift
      ;;
    *)
      LORA_ARGS+=("$1")
      shift
      ;;
  esac
done

python scripts/build_lora_dataset.py "${SYNTH[@]}" --out-dir data/lora/game_text
exec mlx_lm.lora --train -c training/lora_qwen_coder.yaml "${LORA_ARGS[@]}"
