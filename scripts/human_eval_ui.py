#!/usr/bin/env python3
"""
Gradio UI for **human** evaluation of Albert (base or LoRA) on benchmark-style tasks.

Features:
  - Pick a task from a JSON task suite (same shape as `benchmarks/fallen_empire_tasks.json`).
  - Generate a candidate answer with the local MLX model (streaming into the textbox).
  - Score helpfulness / correctness / game-domain fit (1–5) and add free-form notes.
  - Append one JSON line per saved rating to a JSONL file (default under benchmarks/results/).

Run:
  source .venv/bin/activate
  python scripts/human_eval_ui.py
  python scripts/human_eval_ui.py --tasks benchmarks/fallen_empire_tasks.json --output-jsonl benchmarks/results/human_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
DEFAULT_TASKS = Path(__file__).resolve().parent.parent / "benchmarks" / "fallen_empire_tasks.json"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "benchmarks" / "results" / "human_eval.jsonl"
SYSTEM = (
    "You are Albert, assisting with the Fallen Empire strategy game (TypeScript / React / Zustand). "
    "Follow the user instruction literally."
)
LEGACY_UI_DEPRECATION = (
    "scripts/human_eval_ui.py is a legacy UI and is no longer a supported site.\n"
    "Supported sites are:\n"
    "  1) Documentation site: python scripts/private_dashboard_server.py\n"
    "  2) Arena training supervision site: python scripts/game_task_arena.py ui\n"
    "  3) Router prompt site: python scripts/router_chat_gradio.py\n"
    "To run this legacy UI anyway, pass --allow-legacy-ui."
)

if __name__ == "__main__" and "--allow-legacy-ui" not in sys.argv and not any(
    flag in sys.argv for flag in ("-h", "--help")
):
    raise SystemExit(LEGACY_UI_DEPRECATION)

import gradio as gr
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


def _load_task_list(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("tasks file must be a JSON array")
    return data


class Session:
    def __init__(
        self,
        tasks_path: Path,
        model_id: str,
        adapter_path: Optional[str],
        max_tokens: int,
        temp: float,
    ):
        self.tasks_path = tasks_path
        self.tasks = _load_task_list(tasks_path)
        self.by_id = {t["id"]: t for t in self.tasks}
        self.ids = [t["id"] for t in self.tasks]
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.max_tokens = max_tokens
        self.temp = temp
        self.model = None
        self.tokenizer = None

    def ensure_model(self) -> None:
        if self.model is not None:
            return
        load_kw: dict = {}
        if self.adapter_path:
            load_kw["adapter_path"] = self.adapter_path
        self.model, self.tokenizer = load(self.model_id, **load_kw)

    def prompt_for(self, task_id: str) -> str:
        t = self.by_id[task_id]
        return str(t.get("prompt", ""))

    def generate(self, task_id: str) -> str:
        self.ensure_model()
        assert self.tokenizer is not None and self.model is not None
        user = self.prompt_for(task_id)
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        gen_kw: dict = {"max_tokens": self.max_tokens}
        if self.temp > 0:
            gen_kw["sampler"] = make_sampler(temp=self.temp, top_p=1.0)
        acc = ""
        for resp in stream_generate(self.model, self.tokenizer, prompt, **gen_kw):
            acc += resp.text
        return acc

    def save_rating(
        self,
        sink: Path,
        task_id: str,
        response: str,
        h: float,
        c: float,
        g: float,
        notes: str,
    ) -> str:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "task_id": task_id,
            "tasks_file": str(self.tasks_path),
            "model": self.model_id,
            "adapter_path": self.adapter_path,
            "scores": {"helpfulness": h, "correctness": c, "game_domain": g},
            "notes": notes,
            "prompt": self.prompt_for(task_id),
            "response": response,
        }
        sink.parent.mkdir(parents=True, exist_ok=True)
        with sink.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return f"Saved rating for {task_id!r} → {sink}"


def build_ui(sess: Session, out_path: Path, host: str, port: int) -> gr.Blocks:
    with gr.Blocks(title="Albert — human eval") as demo:
        gr.Markdown(
            "## Human evaluation\n"
            f"**Model:** `{sess.model_id}`  \n"
            f"**Tasks:** `{sess.tasks_path}`  \n"
            f"**Adapter:** `{sess.adapter_path or '(none)'}`  \n"
            "Generate a reply, score it, then **Save rating** (appends one JSON line)."
        )
        task_dd = gr.Dropdown(choices=sess.ids, label="Task", value=sess.ids[0] if sess.ids else None)
        prompt_tb = gr.Textbox(label="Prompt (read-only)", lines=6, interactive=False)
        out_tb = gr.Textbox(label="Model output (editable before save)", lines=14)
        gen_btn = gr.Button("Generate")
        h_sl = gr.Slider(1, 5, value=3, step=1, label="Helpfulness")
        c_sl = gr.Slider(1, 5, value=3, step=1, label="Correctness / code quality")
        g_sl = gr.Slider(1, 5, value=3, step=1, label="Game domain fit")
        notes_tb = gr.Textbox(label="Notes (optional)", lines=2)
        save_btn = gr.Button("Save rating")
        status = gr.Markdown("")

        def on_pick(tid: str) -> str:
            return sess.prompt_for(tid) if tid else ""

        task_dd.change(on_pick, inputs=[task_dd], outputs=[prompt_tb])
        demo.load(on_pick, inputs=[task_dd], outputs=[prompt_tb])

        def on_gen(tid: str) -> str:
            if not tid:
                return ""
            return sess.generate(tid)

        gen_btn.click(on_gen, inputs=[task_dd], outputs=[out_tb])

        def on_save(tid: str, response: str, h: float, c: float, g: float, notes: str) -> str:
            if not tid.strip():
                return "Pick a task first."
            msg = sess.save_rating(out_path, tid, response or "", h, c, g, notes or "")
            return msg

        save_btn.click(on_save, inputs=[task_dd, out_tb, h_sl, c_sl, g_sl, notes_tb], outputs=[status])

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Human eval UI for Albert + benchmark tasks")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL))
    parser.add_argument("--adapter-path", default=os.environ.get("ADAPTER_PATH"))
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--allow-legacy-ui",
        action="store_true",
        help="Run this deprecated UI anyway (not a supported site).",
    )
    args = parser.parse_args()
    if not args.allow_legacy_ui:
        raise SystemExit(LEGACY_UI_DEPRECATION)
    print("[deprecated] Running legacy UI: scripts/human_eval_ui.py", file=sys.stderr)

    tp = args.tasks.resolve()
    if not tp.is_file():
        raise SystemExit(f"Tasks file not found: {tp}")

    sess = Session(
        tasks_path=tp,
        model_id=args.model,
        adapter_path=args.adapter_path,
        max_tokens=args.max_tokens,
        temp=args.temp,
    )
    if not sess.ids:
        raise SystemExit(f"No tasks in {tp}")
    demo = build_ui(sess, args.output_jsonl.resolve(), args.host, args.port)
    demo.queue()
    print(f"Human eval UI: http://{args.host}:{args.port}", file=sys.stderr)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
