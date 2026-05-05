---
name: fe-mlx-lab
description: >-
  Repo MLX/Qwen LoRA only—Composer cannot load adapters. Mention @fe-mlx-lab for router/training/smoke ops.
disable-model-invocation: true
---

# MLX lab · save tokens

- **Inference:** `python scripts/router_chat_gradio.py` → `:7864` → accordion **OSS / routing controls** (**Codebase OSS** = `force_route=local`; lock adapter in dropdown). Or `ADAPTER_PATH=… python scripts/chat_gradio.py` (single LoRA).
- **Train/smoke:** `python scripts/fe_ml_lab_runner.py learning` or `python scripts/ml_workflow.py …` in terminal—not step-by-step in chat.
- **Cursor hook (optional):** set **`FE_LAB_CURSOR_HOOK_APPEND=1`** so **`.cursor/hooks.json`** appends **`lab_dashboard/cursor_hook_events.jsonl`** on Agent **`stop`** (see **`lab_dashboard/README.md`**); still no MLX inside Composer.- **Do not paste** long `mlx_lm` logs, manifests, or `run_history.md` tables into Composer; cite **artifact path + run_id** only. Full refs: **`docs/PROJECT_STATE.md`**, **`docs/WORKFLOW.md`** (open only when changing workflow).
