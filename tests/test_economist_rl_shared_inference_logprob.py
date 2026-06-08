"""Shared inference backend must not reload for rollout old_logprob attach."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import economist_rl_ppo_trainer as trainer  # noqa: E402
from economist_rl_ppo_trainer import attach_old_logprob_to_rollout, compute_sequence_logprob  # noqa: E402


def test_compute_sequence_logprob_uses_inference_backend(monkeypatch) -> None:
    backend = MagicMock(name="shared_backend")
    calls: list[MagicMock] = []

    def _fake_from_backend(inference_backend, **kwargs) -> float:
        calls.append(inference_backend)
        return -1.25

    monkeypatch.setattr(trainer, "_sequence_logprob_from_backend", _fake_from_backend)

    lp = compute_sequence_logprob(
        prompt="user task",
        completion="answer text",
        system_prompt="system",
        base_model="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        adapter_path=None,
        inference_backend=backend,
    )
    assert lp == -1.25
    assert calls == [backend]


def test_attach_old_logprob_passes_inference_backend(monkeypatch) -> None:
    backend = MagicMock(name="shared_backend")

    def _fake_compute(**kwargs) -> float:
        assert kwargs.get("inference_backend") is backend
        return -0.5

    monkeypatch.setattr(trainer, "compute_sequence_logprob", _fake_compute)

    row = attach_old_logprob_to_rollout(
        rollout_row={"prompt": "p", "output": "completion"},
        system_prompt="system",
        base_model="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        adapter_path=None,
        inference_backend=backend,
    )
    assert row["old_logprob"] == -0.5


def test_attach_old_logprob_forwards_logprob_window(monkeypatch) -> None:
    backend = MagicMock(name="shared_backend")
    seen: dict[str, object] = {}

    def _fake_compute(**kwargs) -> float:
        seen.update(kwargs)
        return -0.75

    monkeypatch.setattr(trainer, "compute_sequence_logprob", _fake_compute)

    row = attach_old_logprob_to_rollout(
        rollout_row={"prompt": "p", "output": "completion"},
        system_prompt="system",
        base_model="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        adapter_path=None,
        inference_backend=backend,
        max_window_tokens=1024,
    )
    assert row["old_logprob"] == -0.75
    assert seen["max_window_tokens"] == 1024
