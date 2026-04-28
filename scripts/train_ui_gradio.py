#!/usr/bin/env python3
"""
Gradio UI to **prepare data** and **run LoRA training** with live log streaming.

Runs `mlx_lm.lora` as a subprocess (same flags as CLI) so output matches terminal training.
Optional **benchmark** step after a successful train.

Default port **7862** (7860 = chat, 7861 = human eval).

  source .venv/bin/activate
  python scripts/train_ui_gradio.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import gradio as gr

REPO = Path(__file__).resolve().parent.parent
VENV_MLX = REPO / ".venv" / "bin" / "mlx_lm.lora"

# Shared with Stop button
_PROC: Optional[subprocess.Popen] = None


def _mlx_exe() -> str:
    return str(VENV_MLX) if VENV_MLX.is_file() else "mlx_lm.lora"


def stop_training() -> str:
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        _PROC.terminate()
        try:
            _PROC.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _PROC.kill()
        return "Sent stop signal to training process."
    return "No active training process."


def prepare_data(export_path: str, source_repo: str) -> Generator[Tuple[str, str], None, None]:
    """Export + build JSONL; stream combined log."""
    py = sys.executable
    env = {**os.environ, "SOURCE_REPO": source_repo.strip() or os.environ.get("SOURCE_REPO", str(Path.home() / "fallen-empire"))}
    log = ""
    for argv, label in (
        ([py, str(REPO / "scripts" / "export_repo_for_training.py")], "export"),
        (
            [
                py,
                str(REPO / "scripts" / "build_lora_dataset.py"),
                "--from-export",
                export_path.strip() or "data/raw/repo_text.jsonl",
                "--out-dir",
                "data/lora/game_text",
            ],
            "build_lora_dataset",
        ),
    ):
        log += f"\n=== {label} ===\n"
        yield log, "running"
        p = subprocess.run(
            argv,
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
        )
        log += p.stdout or ""
        log += p.stderr or ""
        log += f"\n[exit {p.returncode}]\n"
        yield log, "error" if p.returncode != 0 else "running"
        if p.returncode != 0:
            yield log, f"failed at {label}"
            return
    yield log, "done"


def run_training_job(
    adapter_path: str,
    iters: int,
    batch_size: int,
    max_seq_length: int,
    learning_rate: float,
    grad_checkpoint: bool,
    run_benchmark_after: bool,
) -> Generator[Tuple[str, str], None, None]:
    exe = _mlx_exe()
    argv = [
        exe,
        "--train",
        "-c",
        "training/lora_qwen_coder.yaml",
        "--adapter-path",
        adapter_path.strip() or "checkpoints/fe-lora-ui",
        "--iters",
        str(int(iters)),
        "--batch-size",
        str(int(batch_size)),
        "--max-seq-length",
        str(int(max_seq_length)),
        "--learning-rate",
        str(learning_rate),
        "--val-batches",
        "8",
        "--steps-per-eval",
        "50",
        "--steps-per-report",
        "10",
        "--save-every",
        "100",
    ]
    if grad_checkpoint:
        argv.append("--grad-checkpoint")

    log = ""
    log += f"$ {' '.join(argv)}\n\n"
    yield log, "training"

    global _PROC
    merged = {**os.environ, "PYTHONUNBUFFERED": "1"}
    _PROC = subprocess.Popen(
        argv,
        cwd=str(REPO),
        env=merged,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert _PROC.stdout is not None
    code = 0
    try:
        for line in iter(_PROC.stdout.readline, ""):
            log += line
            yield log, "training"
        _PROC.wait(timeout=60)
        code = _PROC.returncode or 0
    except Exception as e:
        log += f"\n[runner error: {e}]\n"
        code = 1
    finally:
        _PROC = None

    log += f"\n=== train finished exit={code} ===\n"
    yield log, "done" if code == 0 else "error"

    if code == 0 and run_benchmark_after:
        log += "\n=== benchmark ===\n"
        yield log, "benchmark"
        py = sys.executable
        bargv = [
            py,
            str(REPO / "scripts" / "run_game_benchmark.py"),
            "--adapter-path",
            adapter_path.strip() or "checkpoints/fe-lora-ui",
        ]
        bp = subprocess.run(bargv, cwd=str(REPO), capture_output=True, text=True)
        log += bp.stdout or ""
        log += bp.stderr or ""
        log += f"\n[benchmark exit {bp.returncode}]\n"
        yield log, "done" if bp.returncode == 0 else "error"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Fallen Empire — LoRA training") as demo:
        gr.Markdown(
            "## LoRA training UI\n"
            "Prepare **export → JSONL splits**, then run **`mlx_lm.lora`** with live logs. "
            "Use **Stop** if you need to interrupt. After training, optionally run the **benchmark** suite.\n\n"
            "*Tip:* For a documented run (manifest + `docs/run_history.md`), use `python scripts/ml_workflow.py …` from the terminal in parallel or instead."
        )
        with gr.Accordion("1. Data prep", open=True):
            source_repo = gr.Textbox(
                label="SOURCE_REPO (game checkout)",
                value=os.environ.get("SOURCE_REPO", str(Path.home() / "fallen-empire")),
            )
            export_path = gr.Textbox(label="Export JSONL path", value="data/raw/repo_text.jsonl")
            prep_btn = gr.Button("Export + build LoRA dataset")
        with gr.Accordion("2. Training", open=True):
            adapter_path = gr.Textbox(label="Adapter output directory", value="checkpoints/fe-lora-ui")
            iters = gr.Slider(10, 2000, value=300, step=10, label="Iterations")
            batch_size = gr.Slider(1, 8, value=1, step=1, label="Batch size")
            max_seq_length = gr.Slider(512, 8192, value=4096, step=256, label="Max sequence length")
            learning_rate = gr.Number(label="Learning rate", value=1e-5)
            grad_ck = gr.Checkbox(label="Gradient checkpointing", value=True)
            run_bench = gr.Checkbox(label="Run benchmark after successful train", value=True)
            train_btn = gr.Button("Start training", variant="primary")
            stop_btn = gr.Button("Stop training", variant="stop")

        log_box = gr.Textbox(label="Log", lines=28, max_lines=60)
        status = gr.Textbox(label="Status", interactive=False)

        prep_btn.click(
            prepare_data,
            inputs=[export_path, source_repo],
            outputs=[log_box, status],
        )
        train_btn.click(
            run_training_job,
            inputs=[
                adapter_path,
                iters,
                batch_size,
                max_seq_length,
                learning_rate,
                grad_ck,
                run_bench,
            ],
            outputs=[log_box, status],
        )
        stop_btn.click(stop_training, outputs=status)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio LoRA training UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo = build_app()
    demo.queue()
    print(f"Training UI: http://{args.host}:{args.port}", file=sys.stderr)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
