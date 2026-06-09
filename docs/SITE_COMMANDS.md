# Supported site command list

Only the three canonical sites are supported.

## One-time shell setup (new terminal)

```bash
cd /Users/natreed/fallen-empire-lora
source .venv/bin/activate
```

## 1) Documentation site (agent work + cost savings)

Set credentials/tokens before launch (replace placeholders):

```bash
export FE_DASHBOARD_USER=admin
export FE_DASHBOARD_PASSWORD=change-me
export FE_DASHBOARD_INGEST_TOKEN=change-me
python scripts/private_dashboard_server.py
open http://127.0.0.1:8787
```

## 2) Arena training supervision site (specialists vs GPT-5.5/frontier)

```bash
python scripts/game_task_arena.py ui
open http://127.0.0.1:7868
```

Optional model env for frontier lane:

```bash
export FRONTIER_MODEL=gpt-5.5
```

## 3) Prompt router site (agentic specialist routing test)

```bash
python scripts/router_chat_gradio.py
open http://127.0.0.1:7864
```

## Legacy sites

Legacy UIs are intentionally deprecated and blocked by default:

- `scripts/chat_gradio.py`
- `scripts/human_eval_ui.py`
- `scripts/train_ui_gradio.py`
- `scripts/landing_page_arena.py ui`

If you need one for migration/debugging, run it explicitly with `--allow-legacy-ui`.