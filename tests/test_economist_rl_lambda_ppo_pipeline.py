"""Smoke tests for economistRL evidence runner and PPO pipeline wiring."""

from __future__ import annotations

import json
import math
import sys
import unittest
from dataclasses import asdict
from unittest import mock
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


class EconomistRLEvidenceRunnerTests(unittest.TestCase):
    def test_attach_evidence_populates_targeted_tests(self) -> None:
        from economist_rl_evidence_runner import attach_rollout_evidence

        task = {
            "id": "economistRL-market-elasticity-02",
            "subsection": "market_elasticity_pricing",
            "simulation_spec": {
                "tick_count": 20,
                "scenario": {"initial_state": {"currentStock": 50, "desiredStock": 100, "lastPrice": 10, "demand": 1.0}},
                "goals": [
                    {"name": "scarcity_raises_price", "weight": 0.3},
                    {"name": "surplus_lowers_price", "weight": 0.25},
                ],
            },
            "targeted_tests": {"outcome_checks": ["scarcity and surplus cases diverge"]},
        }
        output = (
            "Use stock and desiredStock with bounded clamp smoothing. "
            "Price elasticity curve uses ratio pressure and floor/cap."
        )
        row = attach_rollout_evidence(task=task, rollout_row={"task_id": task["id"], "output": output})
        self.assertEqual(row.get("simulation_source"), "vitest_goals")
        self.assertIn("targeted_tests", row)

    def test_score_after_evidence_has_targeted_tests_component(self) -> None:
        from economist_rl_evidence_runner import attach_rollout_evidence
        from economist_rl_reward_engine import score_output

        payload = json.loads((REPO / "benchmarks" / "economistRL_tasks_v1.json").read_text())
        task = next(t for t in payload["tasks"] if t["id"] == "economistRL-market-elasticity-02")
        output = task["reference_answer"]
        row = attach_rollout_evidence(task=task, rollout_row={"task_id": task["id"], "output": output, "prompt": task["prompt"]})
        row["compiled"] = True
        score = score_output(task, output, payload, rollout_row=row, compiled=True, rolling_compile_rate=0.5)
        self.assertNotIn("simulation_behavior", score["components"])
        self.assertGreater(score["components"]["targeted_tests"], 0.0)


class EconomistRLPPOPipelineTests(unittest.TestCase):
    def test_proxy_old_logprob_uses_mean_token_scale(self) -> None:
        from economist_rl_ppo_trainer import proxy_old_logprob

        short = proxy_old_logprob("one two")
        long = proxy_old_logprob("one two three four five six seven eight nine ten")
        self.assertEqual(short, long)
        self.assertLess(short, 0.0)

    def test_ppo_clip_loss_is_bounded_when_logprobs_match(self) -> None:
        from economist_rl_ppo_trainer import ppo_clip_loss_scalar

        for advantage in (-1.5, -0.2, 0.0, 1.2):
            loss = ppo_clip_loss_scalar(
                new_logprob=-3.0,
                old_logprob=-3.0,
                advantage=advantage,
                clip_epsilon=0.2,
            )
            self.assertAlmostEqual(loss, -advantage, places=6)
            self.assertLess(abs(loss), 10.0)

    def test_mlx_mean_completion_logprob_has_finite_gradients(self) -> None:
        try:
            import mlx.core as mx
        except ImportError:
            self.skipTest("mlx not installed")

        from economist_rl_ppo_trainer import _mean_completion_logprob_mlx

        logits = mx.random.normal((12, 64))
        completion_ids = [3, 7, 11]
        prompt_len = 5

        def loss_fn(x: Any) -> Any:
            return _mean_completion_logprob_mlx(x, completion_ids, prompt_len, mx)

        loss, grads = mx.value_and_grad(loss_fn)(logits)
        self.assertTrue(bool(mx.isfinite(loss).item()))
        self.assertTrue(bool(mx.all(mx.isfinite(grads)).item()))

    def test_mlx_lora_param_key_filter(self) -> None:
        from economist_rl_ppo_trainer import _is_mlx_lora_param_key

        self.assertTrue(_is_mlx_lora_param_key("model.layers.0.mlp.down_proj.lora_a"))
        self.assertFalse(_is_mlx_lora_param_key("model.layers.0.input_layernorm.weight"))

    def test_encode_prompt_and_completion_uses_same_special_token_policy(self) -> None:
        from economist_rl_ppo_trainer import _encode_prompt_and_completion

        class _Tokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "SYS:" + messages[-1]["content"]

            def encode(self, text, add_special_tokens=False):
                self.last_add_special_tokens = add_special_tokens
                return [101, 102, 103]

        tokenizer = _Tokenizer()
        prompt_ids, completion_ids = _encode_prompt_and_completion(
            tokenizer,
            prompt="user task",
            completion="assistant answer",
            system_prompt="system",
        )
        self.assertFalse(tokenizer.last_add_special_tokens)
        self.assertEqual(prompt_ids, [101, 102, 103])
        self.assertEqual(completion_ids, [101, 102, 103])

    def test_completion_logprob_windows_bound_forward_tokens(self) -> None:
        from economist_rl_ppo_trainer import _build_completion_logprob_windows

        windows = _build_completion_logprob_windows(
            prompt_ids=list(range(100)),
            completion_ids=[100, 101, 102, 103],
            max_window_tokens=8,
        )

        self.assertEqual(len(windows), 4)
        self.assertTrue(all(len(window.input_ids) <= 8 for window in windows))
        self.assertEqual(windows[0].input_ids, [93, 94, 95, 96, 97, 98, 99, 100])
        self.assertEqual(windows[0].target_pos, 6)
        self.assertEqual(windows[0].target_token_id, 100)
        self.assertEqual(windows[-1].input_ids, [96, 97, 98, 99, 100, 101, 102, 103])
        self.assertEqual(windows[-1].target_pos, 6)
        self.assertEqual(windows[-1].target_token_id, 103)

    def test_windowed_logprob_matches_full_when_context_fits(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        from economist_rl_ppo_trainer import (
            _build_completion_logprob_windows,
            _mean_completion_logprob_torch,
        )

        prompt_ids = list(range(20))
        completion_ids = [100, 101, 102, 103]
        prompt_len = len(prompt_ids)
        logits = torch.randn(prompt_len + len(completion_ids), 128)

        full_mean = float(
            _mean_completion_logprob_torch(logits, completion_ids, prompt_len, torch)
        )

        window_means = []
        for window in _build_completion_logprob_windows(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            max_window_tokens=512,
        ):
            pos = prompt_len - 1 + completion_ids.index(window.target_token_id)
            log_probs = torch.nn.functional.log_softmax(logits[pos], dim=-1)
            window_means.append(float(log_probs[window.target_token_id]))
        self.assertAlmostEqual(full_mean, sum(window_means) / len(window_means), places=6)

    def test_long_sequence_windowed_path_activates_below_full_context(self) -> None:
        from economist_rl_ppo_trainer import _build_completion_logprob_windows

        prompt_ids = list(range(3000))
        completion_ids = list(range(3000, 3010))
        windows = _build_completion_logprob_windows(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            max_window_tokens=2048,
        )
        self.assertEqual(len(windows), len(completion_ids))
        self.assertTrue(all(len(window.input_ids) <= 2048 for window in windows))
        self.assertEqual(windows[0].input_ids[0], 3000 - 2047)
        self.assertEqual(windows[-1].target_token_id, completion_ids[-1])

    def test_grouped_completion_logprob_windows_cover_tokens_once(self) -> None:
        from economist_rl_ppo_trainer import _build_grouped_completion_logprob_windows

        completion_ids = list(range(3000, 3010))
        windows = _build_grouped_completion_logprob_windows(
            prompt_ids=list(range(3000)),
            completion_ids=completion_ids,
            max_window_tokens=8,
            target_chunk_tokens=3,
        )

        flattened_targets = [token for window in windows for token in window.target_token_ids]
        self.assertEqual(flattened_targets, completion_ids)
        self.assertTrue(all(len(window.input_ids) <= 8 for window in windows))
        self.assertEqual([window.completion_start for window in windows], [0, 3, 6, 9])
        for window in windows:
            first_target_pos = len(window.input_ids) - len(window.target_token_ids) - 1
            self.assertEqual(window.target_positions, list(range(first_target_pos, first_target_pos + len(window.target_token_ids))))

    def test_grouped_runtime_shape_is_far_smaller_than_per_token(self) -> None:
        from economist_rl_ppo_trainer import _build_completion_logprob_windows, _build_grouped_completion_logprob_windows

        prompt_ids = list(range(3000))
        completion_ids = list(range(3000, 6000))
        per_token = _build_completion_logprob_windows(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            max_window_tokens=1536,
        )
        grouped = _build_grouped_completion_logprob_windows(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            max_window_tokens=1536,
            target_chunk_tokens=512,
        )

        self.assertEqual(len(per_token), 3000)
        self.assertLessEqual(len(grouped), 6)
        self.assertTrue(all(len(window.input_ids) <= 1536 for window in grouped))

    def test_grouped_torch_mean_uses_shifted_positions_and_token_weighting(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        from types import SimpleNamespace

        from economist_rl_ppo_trainer import GroupedCompletionLogprobWindow, _mean_grouped_completion_logprob_torch

        log_probs = [
            [math.log(0.1), math.log(0.7), math.log(0.1), math.log(0.05), math.log(0.05)],
            [math.log(0.2), math.log(0.2), math.log(0.2), math.log(0.2), math.log(0.2)],
            [math.log(0.025), math.log(0.025), math.log(0.025), math.log(0.9), math.log(0.025)],
        ]

        class Model:
            def __call__(self, input_ids):
                if input_ids.shape[-1] == 3:
                    return SimpleNamespace(logits=torch.tensor([[log_probs[0], log_probs[1], log_probs[1]]]))
                return SimpleNamespace(logits=torch.tensor([[log_probs[2], log_probs[1]]]))

        windows = [
            GroupedCompletionLogprobWindow(
                input_ids=[10, 11, 12],
                target_positions=[0, 1],
                target_token_ids=[1, 2],
                completion_start=0,
            ),
            GroupedCompletionLogprobWindow(
                input_ids=[12, 13],
                target_positions=[0],
                target_token_ids=[3],
                completion_start=2,
            ),
        ]

        actual = float(_mean_grouped_completion_logprob_torch(model=Model(), windows=windows, torch_mod=torch, device="cpu"))
        token_weighted = (math.log(0.7) + math.log(0.2) + math.log(0.9)) / 3.0
        equal_chunk_weighted = (((math.log(0.7) + math.log(0.2)) / 2.0) + math.log(0.9)) / 2.0

        self.assertAlmostEqual(actual, token_weighted, places=6)
        self.assertNotAlmostEqual(actual, equal_chunk_weighted, places=3)

    def test_transformers_windowed_ppo_backprops_each_window(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import patch

        from economist_rl_ppo_trainer import (
            PPOConfig,
            PPOSample,
            _backward_ppo_sample_transformers,
        )

        class FakeTensor:
            def __init__(self, value: float) -> None:
                self.value = float(value)
                self.dtype = "float32"
                self.device = "cpu"

            def __sub__(self, other):
                other_value = other.value if isinstance(other, FakeTensor) else other
                return FakeTensor(self.value - float(other_value))

            def __mul__(self, other):
                other_value = other.value if isinstance(other, FakeTensor) else other
                return FakeTensor(self.value * float(other_value))

            def __neg__(self):
                return FakeTensor(-self.value)

            def detach(self):
                return self

            def cpu(self):
                return self

            def backward(self) -> None:
                backward_calls.append(self.value)

            def __float__(self) -> float:
                return self.value

        class FakeTorch:
            @staticmethod
            def tensor(value, device=None, dtype=None):
                return FakeTensor(value)

            @staticmethod
            def exp(value):
                return FakeTensor(1.0)

            @staticmethod
            def clamp(value, lo, hi):
                return value

            @staticmethod
            def min(a, b):
                return a if float(a) <= float(b) else b

        backward_calls: list[float] = []
        sample = PPOSample(
            task_id="t1",
            prompt="prompt",
            completion="completion",
            reward=1.0,
            advantage=1.0,
            old_logprob=-1.0,
        )

        with patch(
            "economist_rl_ppo_trainer._windowed_completion_logprob_windows_for_sample",
            return_value=[SimpleNamespace(), SimpleNamespace(), SimpleNamespace()],
        ), patch(
            "economist_rl_ppo_trainer._iter_windowed_completion_logprob_tensors",
            return_value=[FakeTensor(-1.1), FakeTensor(-0.9), FakeTensor(-1.0)],
        ):
            loss_value = _backward_ppo_sample_transformers(
                model=object(),
                tokenizer=object(),
                sample=sample,
                system_prompt="system",
                torch_mod=FakeTorch,
                cfg=PPOConfig(logprob_window_strategy="per_token"),
                device="cpu",
                use_windowed_backward=True,
            )

        self.assertEqual(len(backward_calls), 3)
        self.assertAlmostEqual(loss_value, -1.0, places=6)

    def test_backward_ppo_sample_uses_full_path_when_context_fits(self) -> None:
        from unittest.mock import patch

        from economist_rl_ppo_trainer import (
            PPOConfig,
            PPOSample,
            _backward_ppo_sample_transformers,
        )

        class FakeTensor:
            def __init__(self, value: float) -> None:
                self.value = float(value)
                self.dtype = "float32"
                self.device = "cpu"

            def __sub__(self, other):
                other_value = other.value if isinstance(other, FakeTensor) else other
                return FakeTensor(self.value - float(other_value))

            def __mul__(self, other):
                other_value = other.value if isinstance(other, FakeTensor) else other
                return FakeTensor(self.value * float(other_value))

            def __neg__(self):
                return FakeTensor(-self.value)

            def detach(self):
                return self

            def cpu(self):
                return self

            def backward(self) -> None:
                backward_calls.append(self.value)

            def __float__(self) -> float:
                return self.value

        class FakeTorch:
            @staticmethod
            def tensor(value, device=None, dtype=None):
                return FakeTensor(value)

            @staticmethod
            def exp(value):
                return FakeTensor(1.0)

            @staticmethod
            def clamp(value, lo, hi):
                return value

            @staticmethod
            def min(a, b):
                return a if float(a) <= float(b) else b

        backward_calls: list[float] = []
        sample = PPOSample(
            task_id="t1",
            prompt="prompt",
            completion="completion",
            reward=1.0,
            advantage=1.0,
            old_logprob=-1.0,
        )

        with patch(
            "economist_rl_ppo_trainer._windowed_completion_logprob_windows_for_sample",
            return_value=[],
        ), patch(
            "economist_rl_ppo_trainer._completion_logprob_tensor",
            return_value=FakeTensor(-1.0),
        ) as full_logprob, patch(
            "economist_rl_ppo_trainer._iter_windowed_completion_logprob_tensors"
        ) as windowed_logprobs:
            loss_value = _backward_ppo_sample_transformers(
                model=object(),
                tokenizer=object(),
                sample=sample,
                system_prompt="system",
                torch_mod=FakeTorch,
                cfg=PPOConfig(logprob_window_strategy="per_token"),
                device="cpu",
                use_windowed_backward=None,
            )

        self.assertEqual(len(backward_calls), 1)
        self.assertAlmostEqual(loss_value, -1.0, places=6)
        full_logprob.assert_called_once()
        windowed_logprobs.assert_not_called()

    def test_transformers_grouped_ppo_backprops_once_for_grouped_sample(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import patch

        from economist_rl_ppo_trainer import (
            GroupedCompletionLogprobWindow,
            PPOConfig,
            PPOSample,
            _backward_ppo_sample_transformers,
        )

        class FakeTensor:
            def __init__(self, value: float) -> None:
                self.value = float(value)
                self.dtype = "float32"
                self.device = "cpu"

            def __sub__(self, other):
                other_value = other.value if isinstance(other, FakeTensor) else other
                return FakeTensor(self.value - float(other_value))

            def __mul__(self, other):
                other_value = other.value if isinstance(other, FakeTensor) else other
                return FakeTensor(self.value * float(other_value))

            def __neg__(self):
                return FakeTensor(-self.value)

            def detach(self):
                return self

            def cpu(self):
                return self

            def backward(self) -> None:
                backward_calls.append(self.value)

            def __float__(self) -> float:
                return self.value

        class FakeTorch:
            @staticmethod
            def tensor(value, device=None, dtype=None):
                return FakeTensor(value)

            @staticmethod
            def exp(value):
                return FakeTensor(1.0)

            @staticmethod
            def clamp(value, lo, hi):
                return value

            @staticmethod
            def min(a, b):
                return a if float(a) <= float(b) else b

        backward_calls: list[float] = []
        sample = PPOSample(
            task_id="t1",
            prompt="prompt",
            completion="completion",
            reward=1.0,
            advantage=1.0,
            old_logprob=-1.0,
        )
        grouped_windows = [
            GroupedCompletionLogprobWindow(
                input_ids=[1, 2, 3],
                target_positions=[1],
                target_token_ids=[3],
                completion_start=0,
            )
        ]

        with patch(
            "economist_rl_ppo_trainer._grouped_completion_logprob_windows_for_sample",
            return_value=grouped_windows,
        ), patch(
            "economist_rl_ppo_trainer._mean_grouped_completion_logprob_torch",
            return_value=FakeTensor(-1.0),
        ) as grouped_mean, patch(
            "economist_rl_ppo_trainer._iter_windowed_completion_logprob_tensors"
        ) as per_token_iter:
            loss_value = _backward_ppo_sample_transformers(
                model=object(),
                tokenizer=object(),
                sample=sample,
                system_prompt="system",
                torch_mod=FakeTorch,
                cfg=PPOConfig(logprob_window_strategy="grouped", target_logprob_chunk_tokens=512),
                device="cpu",
                use_windowed_backward=True,
            )

        self.assertEqual(len(backward_calls), 1)
        self.assertAlmostEqual(loss_value, -1.0, places=6)
        grouped_mean.assert_called_once()
        per_token_iter.assert_not_called()

    def test_transformers_4bit_requested_for_mlx_4bit_cuda_model(self) -> None:
        from economist_rl_ppo_trainer import _should_load_transformers_4bit

        self.assertTrue(
            _should_load_transformers_4bit(
                "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
                "cuda",
            )
        )
        self.assertFalse(
            _should_load_transformers_4bit(
                "Qwen/Qwen2.5-Coder-7B-Instruct",
                "cuda",
            )
        )
        self.assertFalse(
            _should_load_transformers_4bit(
                "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
                "cpu",
            )
        )

    def test_build_ppo_samples_and_dry_run_train(self) -> None:
        from economist_rl_ppo_trainer import PPOConfig, build_ppo_samples, train_ppo_batch

        scored_rows = [
            {
                "task_id": "t1",
                "rollout": {"prompt": "prompt one", "output": "answer one", "old_logprob": -3.0},
                "score": {"reward": 0.8, "score": 80.0},
            },
            {
                "task_id": "t2",
                "rollout": {"prompt": "prompt two", "output": "answer two", "old_logprob": -2.5},
                "score": {"reward": 0.2, "score": 20.0},
            },
        ]
        samples = build_ppo_samples(scored_rows=scored_rows, system_prompt="system")
        self.assertEqual(len(samples), 2)
        progress_events: list[dict[str, Any]] = []
        manifest = train_ppo_batch(
            scored_rows=scored_rows,
            system_prompt="system",
            base_model="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
            source_adapter=REPO / "checkpoints" / "adapters" / "economistRL" / "seed_bootstrap",
            candidate_adapter=REPO / "checkpoints" / "adapters" / "economistRL" / "_ppo_test_candidate",
            cfg=PPOConfig(min_samples=2),
            dry_run=True,
            train_backend="mlx",
            progress_callback=progress_events.append,
        )
        self.assertEqual(manifest["status"], "dry_run_no_weight_update")
        self.assertEqual(manifest["train_backend"], "mlx")
        self.assertEqual([event["phase"] for event in progress_events], ["start", "complete"])

    def test_build_ppo_samples_caps_completion_outliers_by_batch_ratio(self) -> None:
        from economist_rl_ppo_trainer import PPOConfig, build_ppo_samples

        scored_rows = [
            {
                "task_id": "short-good",
                "rollout": {"prompt": "prompt", "output": "a" * 100, "old_logprob": -1.0},
                "score": {"reward": 0.8, "training_usable": True},
            },
            {
                "task_id": "short-bad",
                "rollout": {"prompt": "prompt", "output": "b" * 110, "old_logprob": -1.0},
                "score": {"reward": 0.1, "training_usable": True},
            },
            {
                "task_id": "long-zero",
                "rollout": {"prompt": "prompt", "output": "c" * 1000, "old_logprob": -1.0},
                "score": {"reward": 0.0, "training_usable": True},
            },
        ]

        samples = build_ppo_samples(
            scored_rows=scored_rows,
            system_prompt="system",
            cfg=PPOConfig(completion_outlier_cap_ratio=2.0),
        )

        self.assertEqual([sample.task_id for sample in samples], ["short-good", "short-bad", "long-zero"])
        long = samples[-1]
        self.assertEqual(long.reward, 0.0)
        self.assertTrue(long.completion_capped)
        self.assertEqual(long.original_completion_chars, 1000)
        self.assertEqual(len(long.completion), 220)

    def test_build_ppo_samples_can_disable_completion_outlier_cap(self) -> None:
        from economist_rl_ppo_trainer import PPOConfig, build_ppo_samples

        scored_rows = [
            {
                "task_id": f"t{i}",
                "rollout": {"prompt": "prompt", "output": "x" * length, "old_logprob": -1.0},
                "score": {"reward": 0.5, "training_usable": True},
            }
            for i, length in enumerate([100, 110, 1000])
        ]

        samples = build_ppo_samples(
            scored_rows=scored_rows,
            system_prompt="system",
            cfg=PPOConfig(completion_outlier_cap_ratio=0.0),
        )

        self.assertEqual(len(samples[-1].completion), 1000)
        self.assertFalse(samples[-1].completion_capped)

    def test_train_ppo_batch_uses_transformers_backend_when_forced(self) -> None:
        from economist_rl_ppo_trainer import PPOConfig, train_ppo_batch

        scored_rows = [
            {
                "task_id": "t1",
                "rollout": {"prompt": "prompt one", "output": "answer one", "old_logprob": -3.0},
                "score": {"reward": 0.8, "score": 80.0},
            },
            {
                "task_id": "t2",
                "rollout": {"prompt": "prompt two", "output": "answer two", "old_logprob": -2.5},
                "score": {"reward": 0.2, "score": 20.0},
            },
        ]
        manifest = train_ppo_batch(
            scored_rows=scored_rows,
            system_prompt="system",
            base_model="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
            source_adapter=REPO / "checkpoints" / "adapters" / "economistRL" / "seed_bootstrap",
            candidate_adapter=REPO / "checkpoints" / "adapters" / "economistRL" / "_ppo_test_candidate_tf",
            cfg=PPOConfig(min_samples=2),
            dry_run=True,
            train_backend="transformers",
        )
        self.assertEqual(manifest["train_backend"], "transformers")
        self.assertEqual(manifest["status"], "dry_run_no_weight_update")

    def test_ppo_subprocess_writes_progress_sidecar(self) -> None:
        import tempfile

        from economist_rl_ppo_trainer import PPOConfig

        sys.path.insert(0, str(REPO / "scripts" / "lambda"))
        from run_economist_rl_ppo_train import run_ppo_train_from_request

        scored_rows = [
            {
                "task_id": "t1",
                "rollout": {"prompt": "prompt one", "output": "answer one", "old_logprob": -3.0},
                "score": {"reward": 0.8, "score": 80.0},
            },
            {
                "task_id": "t2",
                "rollout": {"prompt": "prompt two", "output": "answer two", "old_logprob": -2.5},
                "score": {"reward": 0.2, "score": 20.0},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scored_file = root / "scored.jsonl"
            manifest_file = root / "ppo" / "ppo_train_001.json"
            progress_file = root / "ppo" / "ppo_train_001.progress.json"
            scored_file.write_text("\n".join(json.dumps(row) for row in scored_rows) + "\n", encoding="utf-8")

            manifest = run_ppo_train_from_request(
                {
                    "scored_file": str(scored_file),
                    "ppo_manifest_file": str(manifest_file),
                    "progress_file": str(progress_file),
                    "source_adapter": str(root / "source"),
                    "candidate_adapter": str(root / "candidate"),
                    "base_model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
                    "system_prompt": "system",
                    "ppo_config": asdict(PPOConfig(min_samples=2)),
                    "train_backend": "transformers",
                    "dry_run": True,
                }
            )

            progress = json.loads(progress_file.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "dry_run_no_weight_update")
            self.assertEqual(manifest["progress_file"], str(progress_file.resolve()))
            self.assertEqual(progress["schema_version"], "economist_rl_ppo_progress_v1")
            self.assertEqual(progress["phase"], "manifest_written")
            self.assertEqual(progress["status"], "dry_run_no_weight_update")


class EconomistRLAdapterChainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import importlib.util
        import sys

        module_path = REPO / "scripts" / "lambda" / "run_economist_rl_lambda_cycle.py"
        spec = importlib.util.spec_from_file_location("run_economist_rl_lambda_cycle", module_path)
        cycle = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules["run_economist_rl_lambda_cycle"] = cycle
        spec.loader.exec_module(cycle)
        cls.cycle = cycle

    def test_cycle_two_chains_from_previous_candidate_when_registry_unchanged(self) -> None:
        import json
        import tempfile

        from economist_rl_peft import build_peft_adapter_config
        from economist_rl_ppo_trainer import mark_candidate_ppo_trained

        registry = self.cycle.resolve_registry_adapter(
            REPO / "training" / "adapter_registry_v1.json",
            "economistRL",
        )
        with tempfile.TemporaryDirectory() as tmp:
            prev = Path(tmp) / "rl_pass_003"
            prev.mkdir()
            (prev / "adapter_model.safetensors").write_bytes(b"")
            (prev / "adapter_config.json").write_text(
                json.dumps(
                    build_peft_adapter_config(
                        base_model_name_or_path="Qwen/Qwen2.5-Coder-7B-Instruct",
                        target_modules=["q_proj"],
                        rank=16,
                        lora_alpha=20.0,
                        lora_dropout=0.05,
                    )
                ),
                encoding="utf-8",
            )
            mark_candidate_ppo_trained(prev, manifest={"status": "trained"})
            reg, effective, reason = self.cycle.resolve_cycle_current_adapter(
                registry_path=REPO / "training" / "adapter_registry_v1.json",
                adapter_name="economistRL",
                chain_from_previous=prev,
            )
        self.assertEqual(reg.adapter_path, registry.adapter_path)
        self.assertEqual(reason, "previous_cycle_candidate")
        self.assertTrue(str(effective.resolved_path).endswith("rl_pass_003"))

    def test_chain_skipped_when_same_as_registry(self) -> None:
        from economist_rl_ppo_trainer import mark_candidate_ppo_trained

        registry = self.cycle.resolve_registry_adapter(
            REPO / "training" / "adapter_registry_v1.json",
            "economistRL",
        )
        if not registry.exists:
            self.skipTest(f"registry adapter missing: {registry.resolved_path}")
        mark_candidate_ppo_trained(registry.resolved_path, manifest={"status": "trained"})
        reg, effective, reason = self.cycle.resolve_cycle_current_adapter(
            registry_path=REPO / "training" / "adapter_registry_v1.json",
            adapter_name="economistRL",
            chain_from_previous=registry.resolved_path,
        )
        self.assertEqual(reason, "registry")
        self.assertEqual(reg.resolved_path, effective.resolved_path)

    def test_untrained_chain_is_ignored(self) -> None:
        import json
        import tempfile

        from economist_rl_peft import build_peft_adapter_config

        with tempfile.TemporaryDirectory() as tmp:
            prev = Path(tmp) / "rl_pass_copy"
            prev.mkdir()
            (prev / "adapter_model.safetensors").write_bytes(b"")
            (prev / "adapter_config.json").write_text(
                json.dumps(
                    build_peft_adapter_config(
                        base_model_name_or_path="Qwen/Qwen2.5-Coder-7B-Instruct",
                        target_modules=["q_proj"],
                        rank=16,
                        lora_alpha=20.0,
                        lora_dropout=0.05,
                    )
                ),
                encoding="utf-8",
            )
            _reg, effective, reason = self.cycle.resolve_cycle_current_adapter(
                registry_path=REPO / "training" / "adapter_registry_v1.json",
                adapter_name="economistRL",
                chain_from_previous=prev,
            )
        self.assertEqual(reason, "registry_missing_chain_weights")
        self.assertEqual(_reg.resolved_path, effective.resolved_path)

    def test_chain_from_peft_only_candidate(self) -> None:
        import json
        import tempfile

        from economist_rl_peft import build_peft_adapter_config
        from economist_rl_ppo_trainer import mark_candidate_ppo_trained

        with tempfile.TemporaryDirectory() as tmp:
            prev = Path(tmp) / "rl_pass_peft"
            prev.mkdir()
            (prev / "adapter_model.safetensors").write_bytes(b"")
            (prev / "adapter_config.json").write_text(
                json.dumps(
                    build_peft_adapter_config(
                        base_model_name_or_path="Qwen/Qwen2.5-Coder-7B-Instruct",
                        target_modules=["q_proj"],
                        rank=16,
                        lora_alpha=20.0,
                        lora_dropout=0.05,
                    )
                ),
                encoding="utf-8",
            )
            mark_candidate_ppo_trained(prev, manifest={"status": "trained"})
            reg, effective, reason = self.cycle.resolve_cycle_current_adapter(
                registry_path=REPO / "training" / "adapter_registry_v1.json",
                adapter_name="economistRL",
                chain_from_previous=prev,
            )
            self.assertEqual(reason, "previous_cycle_candidate")
            self.assertEqual(effective.resolved_path, prev.resolve())
            self.assertTrue(effective.exists)
            self.assertNotEqual(reg.resolved_path, effective.resolved_path)

    def test_init_adapter_path_yields_to_chain_from_previous(self) -> None:
        import json
        import tempfile

        from economist_rl_peft import build_peft_adapter_config
        from economist_rl_ppo_trainer import mark_candidate_ppo_trained

        with tempfile.TemporaryDirectory() as tmp:
            prev = Path(tmp) / "rl_pass_peft"
            prev.mkdir()
            (prev / "adapter_model.safetensors").write_bytes(b"")
            (prev / "adapter_config.json").write_text(
                json.dumps(
                    build_peft_adapter_config(
                        base_model_name_or_path="Qwen/Qwen2.5-Coder-7B-Instruct",
                        target_modules=["q_proj"],
                        rank=16,
                        lora_alpha=20.0,
                        lora_dropout=0.05,
                    )
                ),
                encoding="utf-8",
            )
            mark_candidate_ppo_trained(prev, manifest={"status": "trained"})
            init = REPO / "checkpoints" / "fe-lora-arena-apply-sft"
            reg, effective, reason = self.cycle.resolve_cycle_current_adapter(
                registry_path=REPO / "training" / "adapter_registry_v1.json",
                adapter_name="economistRL",
                chain_from_previous=prev,
                init_adapter_path=init,
            )
            self.assertEqual(reason, "previous_cycle_candidate")
            self.assertEqual(effective.resolved_path, prev.resolve())


class EconomistRLSubprocessPPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import importlib.util
        import sys

        module_path = REPO / "scripts" / "lambda" / "run_economist_rl_lambda_cycle.py"
        spec = importlib.util.spec_from_file_location("run_economist_rl_lambda_cycle", module_path)
        cycle = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules["run_economist_rl_lambda_cycle"] = cycle
        spec.loader.exec_module(cycle)
        cls.cycle = cycle

    def test_run_ppo_train_subprocess_invokes_child_with_request_file(self) -> None:
        import tempfile
        from unittest.mock import patch

        scored_rows = [
            {
                "task_id": "t1",
                "rollout": {"prompt": "p", "output": "o", "old_logprob": -1.0},
                "score": {"reward": 0.5, "score": 50.0},
            },
            {
                "task_id": "t2",
                "rollout": {"prompt": "p2", "output": "o2", "old_logprob": -2.0},
                "score": {"reward": 0.4, "score": 40.0},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scored_file = root / "scored_batch_001.jsonl"
            scored_file.write_text(
                "\n".join(json.dumps(row) for row in scored_rows) + "\n",
                encoding="utf-8",
            )
            ppo_manifest_file = root / "ppo" / "ppo_train_001.json"
            source_adapter = REPO / "checkpoints" / "adapters" / "economistRL" / "seed_bootstrap"
            candidate_adapter = root / "rl_pass_001"

            def _fake_run(cmd, cwd, env, check, stdout=None, stderr=None):
                request_path = Path(cmd[cmd.index("--request") + 1])
                self.assertTrue(request_path.is_file())
                request = json.loads(request_path.read_text(encoding="utf-8"))
                self.assertEqual(request["scored_file"], str(scored_file.resolve()))
                self.assertEqual(request["train_backend"], "transformers")
                if stderr is not None:
                    stderr.write("subprocess ok\n")
                ppo_manifest_file.parent.mkdir(parents=True, exist_ok=True)
                ppo_manifest_file.write_text(
                    json.dumps({"status": "trained", "train_backend": "transformers"}) + "\n",
                    encoding="utf-8",
                )
                from types import SimpleNamespace

                return SimpleNamespace(returncode=0)

            with patch.object(self.cycle.subprocess, "run", side_effect=_fake_run):
                manifest = self.cycle.run_ppo_train_subprocess(
                    scored_file=scored_file,
                    system_prompt="system",
                    base_model="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
                    source_adapter=source_adapter,
                    candidate_adapter=candidate_adapter,
                    ppo_manifest_file=ppo_manifest_file,
                    ppo_config=self.cycle.PPOConfig(min_samples=2),
                    train_backend="transformers",
                    dry_run=False,
                )
            self.assertEqual(manifest["status"], "trained")
            self.assertTrue(manifest.get("subprocess_isolated"))
            self.assertIn("ppo_stderr_log", manifest)

    def test_run_ppo_train_subprocess_failure_captures_logs_and_discards_candidate(self) -> None:
        import tempfile
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scored_file = root / "scored_batch_001.jsonl"
            scored_file.write_text("{}\n", encoding="utf-8")
            ppo_manifest_file = root / "ppo" / "ppo_train_001.json"
            source_adapter = root / "source"
            source_adapter.mkdir()
            candidate_adapter = root / "rl_pass_001"
            candidate_adapter.mkdir()
            (candidate_adapter / "adapter_model.safetensors").write_bytes(b"stale")

            def _fake_run(cmd, cwd, env, check, stdout=None, stderr=None):
                if stderr is not None:
                    stderr.write("CUDA out of memory\n")
                from types import SimpleNamespace

                return SimpleNamespace(returncode=-9)

            with patch.object(self.cycle.subprocess, "run", side_effect=_fake_run):
                manifest = self.cycle.run_ppo_train_subprocess(
                    scored_file=scored_file,
                    system_prompt="system",
                    base_model="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
                    source_adapter=source_adapter,
                    candidate_adapter=candidate_adapter,
                    ppo_manifest_file=ppo_manifest_file,
                    ppo_config=self.cycle.PPOConfig(min_samples=1),
                    train_backend="transformers",
                    dry_run=False,
                )
            self.assertEqual(manifest["reason"], "subprocess_exit")
            self.assertEqual(manifest["exit_code"], -9)
            self.assertIn("stderr_tail", manifest)
            self.assertFalse(candidate_adapter.exists())
            stderr_log = Path(manifest["ppo_stderr_log"])
            self.assertTrue(stderr_log.is_file())
            self.assertIn("CUDA out of memory", stderr_log.read_text(encoding="utf-8"))

    def test_ppo_train_child_dry_run_reads_scored_jsonl(self) -> None:
        import importlib.util
        import sys
        import tempfile

        module_path = REPO / "scripts" / "lambda" / "run_economist_rl_ppo_train.py"
        spec = importlib.util.spec_from_file_location("run_economist_rl_ppo_train", module_path)
        child = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules["run_economist_rl_ppo_train"] = child
        spec.loader.exec_module(child)

        scored_rows = [
            {
                "task_id": "t1",
                "rollout": {"prompt": "p", "output": "o", "old_logprob": -1.0},
                "score": {"reward": 0.5, "score": 50.0},
            },
            {
                "task_id": "t2",
                "rollout": {"prompt": "p2", "output": "o2", "old_logprob": -2.0},
                "score": {"reward": 0.4, "score": 40.0},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scored_file = root / "scored.jsonl"
            scored_file.write_text(
                "\n".join(json.dumps(row) for row in scored_rows) + "\n",
                encoding="utf-8",
            )
            ppo_manifest_file = root / "ppo_train.json"
            request = {
                "scored_file": str(scored_file),
                "ppo_manifest_file": str(ppo_manifest_file),
                "system_prompt": "system",
                "base_model": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
                "source_adapter": str(REPO / "checkpoints" / "adapters" / "economistRL" / "seed_bootstrap"),
                "candidate_adapter": str(root / "candidate"),
                "ppo_config": {"min_samples": 2},
                "train_backend": "mlx",
                "dry_run": True,
            }
            manifest = child.run_ppo_train_from_request(request)
            self.assertEqual(manifest["status"], "dry_run_no_weight_update")
            self.assertTrue(ppo_manifest_file.is_file())


class EconomistRLLambdaCycleDryRunTests(unittest.TestCase):
    def test_default_ppo_config_prefers_worker_split_batch_sizes(self) -> None:
        import argparse
        import importlib.util
        import sys

        module_path = REPO / "scripts" / "lambda" / "run_economist_rl_lambda_cycle.py"
        spec = importlib.util.spec_from_file_location("run_economist_rl_lambda_cycle", module_path)
        cycle = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules["run_economist_rl_lambda_cycle"] = cycle
        spec.loader.exec_module(cycle)

        args = argparse.Namespace(
            ppo_epochs=None,
            ppo_min_samples=None,
            ppo_max_samples=None,
            ppo_logprob_window_tokens=None,
            ppo_mini_batch_size=None,
            dry_run=False,
        )
        cfg = cycle._resolve_ppo_config(cycle.SPECIALIZATIONS["economist_rl"], args)
        self.assertEqual(cfg.min_samples, 25)
        self.assertEqual(cfg.max_samples, 64)

        args.ppo_mini_batch_size = 16
        cfg = cycle._resolve_ppo_config(cycle.SPECIALIZATIONS["economist_rl"], args)
        self.assertEqual(cfg.mini_batch_size, 16)

    def test_rollout_limit_uses_bootstrap_only_for_first_worker_cycle(self) -> None:
        import argparse
        import importlib.util
        import sys

        module_path = REPO / "scripts" / "lambda" / "run_economist_rl_lambda_cycle.py"
        spec = importlib.util.spec_from_file_location("run_economist_rl_lambda_cycle", module_path)
        cycle = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules["run_economist_rl_lambda_cycle"] = cycle
        spec.loader.exec_module(cycle)

        args = argparse.Namespace(rollouts_per_cycle=25, bootstrap_rollouts_per_cycle=50)
        self.assertEqual(cycle._rollout_limit_for_cycle(args, bootstrap_cycle=True), 50)
        self.assertEqual(cycle._rollout_limit_for_cycle(args, bootstrap_cycle=False), 25)

    def test_watcher_oom_backoff_keeps_ppo_max_at_or_above_min(self) -> None:
        import importlib.util
        import sys

        module_path = REPO / "scripts" / "lambda" / "watch_economist_rl_lambda_overnight.py"
        spec = importlib.util.spec_from_file_location("watch_economist_rl_lambda_overnight", module_path)
        watcher = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules["watch_economist_rl_lambda_overnight"] = watcher
        spec.loader.exec_module(watcher)

        argv = [
            "--ppo-min-samples",
            "25",
            "--ppo-max-samples",
            "25",
            "--ppo-logprob-window-tokens",
            "1536",
        ]
        with mock.patch.object(watcher, "_log"):
            adapted = watcher._adapt_cycle_argv_after_failure(
                argv,
                progress={"failure_hits": ["training_failed"]},
                remote_log="cuda out of memory",
            )
        self.assertEqual(adapted[adapted.index("--ppo-max-samples") + 1], "25")

    def test_dry_run_cycle_manifest_uses_ppo_path(self) -> None:
        import argparse
        import importlib.util
        import sys

        module_path = REPO / "scripts" / "lambda" / "run_economist_rl_lambda_cycle.py"
        spec = importlib.util.spec_from_file_location("run_economist_rl_lambda_cycle", module_path)
        cycle = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules["run_economist_rl_lambda_cycle"] = cycle
        spec.loader.exec_module(cycle)

        args = argparse.Namespace(
            specialization="economist_rl",
            adapter_name=None,
            task_db=None,
            eval_set=None,
            output_root=REPO / "benchmarks" / "results" / "economistRL" / "_dry_run_out",
            results_root=REPO / "benchmarks" / "results" / "economistRL" / "_dry_run_results",
            data_root=REPO / "data" / "economistRL" / "_dry_run_data",
            registry=REPO / "training" / "adapter_registry_v1.json",
            base_model=cycle.DEFAULT_BASE_MODEL,
            rollouts_per_cycle=2,
            bootstrap_rollouts_per_cycle=None,
            ppo_epochs=None,
            ppo_min_samples=1,
            max_tokens=64,
            temperature=0.0,
            eval_limit=1,
            dry_run=True,
            lambda_mode=False,
            train_config=None,
            cycles=1,
            skip_execution=True,
            execution_timeout_s=600,
            execution_compile_command=[],
            no_require_execution_source=True,
            skip_eval=False,
            skip_ppo=False,
            strict_old_logprob=False,
            context_max_chars=12_000,
            eval_manifest=None,
            init_adapter_path=None,
            ppo_max_samples=None,
            ppo_logprob_window_tokens=None,
            ppo_in_process=False,
            execution_source_repo=None,
        )
        cycle.apply_specialization_defaults(args)
        manifest = cycle.run_cycle(args, cycle_id=999)
        self.assertEqual(manifest["train_mode"], "ppo")
        self.assertIn("evidence_file", manifest)
        self.assertIn("ppo_manifest_file", manifest)
        self.assertEqual(manifest["cycle_status"], "dry_run")
        self.assertFalse(manifest.get("registry_auto_update"))
        eval_result = manifest.get("eval_result") or {}
        self.assertEqual(eval_result.get("status"), "skipped")


if __name__ == "__main__":
    unittest.main()
