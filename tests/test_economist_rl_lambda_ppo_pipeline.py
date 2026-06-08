"""Smoke tests for economistRL evidence runner and PPO pipeline wiring."""

from __future__ import annotations

import json
import sys
import unittest
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
        manifest = train_ppo_batch(
            scored_rows=scored_rows,
            system_prompt="system",
            base_model="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
            source_adapter=REPO / "checkpoints" / "adapters" / "economistRL" / "seed_bootstrap",
            candidate_adapter=REPO / "checkpoints" / "adapters" / "economistRL" / "_ppo_test_candidate",
            cfg=PPOConfig(min_samples=2),
            dry_run=True,
            train_backend="mlx",
        )
        self.assertEqual(manifest["status"], "dry_run_no_weight_update")
        self.assertEqual(manifest["train_backend"], "mlx")

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
        registry = self.cycle.resolve_registry_adapter(
            REPO / "training" / "adapter_registry_v1.json",
            "economistRL",
        )
        prev = REPO / "checkpoints" / "adapters" / "economistRL" / "rl_pass_003"
        reg, effective, reason = self.cycle.resolve_cycle_current_adapter(
            registry_path=REPO / "training" / "adapter_registry_v1.json",
            adapter_name="economistRL",
            chain_from_previous=prev,
        )
        self.assertEqual(reg.adapter_path, registry.adapter_path)
        self.assertEqual(reason, "previous_cycle_candidate")
        self.assertTrue(str(effective.resolved_path).endswith("rl_pass_003"))

    def test_chain_skipped_when_same_as_registry(self) -> None:
        seed = REPO / "checkpoints" / "adapters" / "economistRL" / "seed_bootstrap"
        reg, effective, reason = self.cycle.resolve_cycle_current_adapter(
            registry_path=REPO / "training" / "adapter_registry_v1.json",
            adapter_name="economistRL",
            chain_from_previous=seed,
        )
        self.assertEqual(reason, "registry")
        self.assertEqual(reg.resolved_path, effective.resolved_path)

    def test_chain_from_peft_only_candidate(self) -> None:
        import json
        import tempfile

        from economist_rl_peft import build_peft_adapter_config

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
            reg, effective, reason = self.cycle.resolve_cycle_current_adapter(
                registry_path=REPO / "training" / "adapter_registry_v1.json",
                adapter_name="economistRL",
                chain_from_previous=prev,
            )
            self.assertEqual(reason, "previous_cycle_candidate")
            self.assertEqual(effective.resolved_path, prev.resolve())
            self.assertTrue(effective.exists)
            self.assertNotEqual(reg.resolved_path, effective.resolved_path)


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

            def _fake_run(cmd, cwd, env, check):
                request_path = Path(cmd[cmd.index("--request") + 1])
                self.assertTrue(request_path.is_file())
                request = json.loads(request_path.read_text(encoding="utf-8"))
                self.assertEqual(request["scored_file"], str(scored_file.resolve()))
                self.assertEqual(request["train_backend"], "transformers")
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
            ppo_epochs=None,
            ppo_min_samples=1,
            max_tokens=64,
            temperature=0.0,
            eval_limit=1,
            dry_run=True,
            lambda_mode=False,
            train_config=None,
            cycles=1,
        )
        cycle.apply_specialization_defaults(args)
        manifest = cycle.run_cycle(args, cycle_id=999)
        self.assertEqual(manifest["train_mode"], "ppo")
        self.assertIn("evidence_file", manifest)
        self.assertIn("ppo_manifest_file", manifest)
        self.assertEqual(manifest["cycle_status"], "dry_run")
        self.assertFalse(manifest.get("registry_auto_update"))
        eval_result = manifest.get("eval_result") or {}
        comparisons = eval_result.get("comparisons") or {}
        self.assertIn("vs_registry_baseline", comparisons)
        self.assertIn("vs_working_source", comparisons)
        self.assertIn("decision", comparisons["vs_registry_baseline"])
        self.assertIn("decision", comparisons["vs_working_source"])


if __name__ == "__main__":
    unittest.main()
