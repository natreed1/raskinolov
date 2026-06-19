"""Tests for economistRL split-worker queue/state mechanics."""

from __future__ import annotations

import json
import argparse
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "lambda"))


class EconomistRlSplitWorkerOrchestratorTests(unittest.TestCase):
    def test_enqueue_training_usable_samples_filters_and_tags_adapter_version(self) -> None:
        from run_economist_rl_split_workers import enqueue_training_usable_samples

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scored_file = root / "scored.jsonl"
            queue_file = root / "queue.jsonl"
            scored_rows = [
                {"task_id": "good", "score": {"training_usable": True, "reward": 0.7}},
                {"task_id": "empty", "score": {"training_usable": False, "reward": 0.0}},
                {"task_id": "implicit-good", "score": {"reward": 0.4}},
            ]
            scored_file.write_text("\n".join(json.dumps(row) for row in scored_rows) + "\n", encoding="utf-8")

            appended = enqueue_training_usable_samples(
                scored_file=scored_file,
                queue_file=queue_file,
                adapter_version="seed",
                rollout_cycle_id=3,
            )

            self.assertEqual(appended, 2)
            queued = [json.loads(line) for line in queue_file.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["task_id"] for row in queued], ["good", "implicit-good"])
            self.assertTrue(all(row["adapter_version"] == "seed" for row in queued))
            self.assertTrue(all(row["rollout_cycle_id"] == 3 for row in queued))

    def test_select_unconsumed_samples_advances_offset_and_caps_batch(self) -> None:
        from run_economist_rl_split_workers import select_unconsumed_samples

        with tempfile.TemporaryDirectory() as tmp:
            queue_file = Path(tmp) / "queue.jsonl"
            rows = [{"task_id": f"t{i}", "score": {"reward": i / 10}} for i in range(5)]
            queue_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            selected, next_offset = select_unconsumed_samples(queue_file=queue_file, offset=1, max_samples=3)

            self.assertEqual([row["task_id"] for row in selected], ["t1", "t2", "t3"])
            self.assertEqual(next_offset, 4)

    def test_promote_candidate_updates_latest_adapter_state(self) -> None:
        from run_economist_rl_split_workers import SplitWorkerState, promote_candidate

        state = SplitWorkerState(
            latest_approved_adapter="/adapters/seed",
            latest_adapter_version="seed",
            queue_read_offset=4,
            ppo_updates=1,
        )
        manifest = {
            "status": "trained",
            "candidate_adapter": "/adapters/rl_pass_002",
            "sample_stats": {"mean_reward": 0.55},
        }

        promoted = promote_candidate(state, manifest, acceptance_mode="trained_marker")

        self.assertEqual(promoted.latest_approved_adapter, "/adapters/rl_pass_002")
        self.assertEqual(promoted.latest_adapter_version, "rl_pass_002")
        self.assertEqual(promoted.ppo_updates, 2)
        self.assertEqual(promoted.last_train_status, "trained")

    def test_ppo_config_from_args_includes_mini_batch_size(self) -> None:
        from run_economist_rl_split_workers import _ppo_config_from_args

        args = argparse.Namespace(
            ppo_epochs=1,
            ppo_min_samples=25,
            ppo_max_samples=64,
            ppo_logprob_window_tokens=1536,
            ppo_logprob_window_strategy="grouped",
            ppo_target_logprob_chunk_tokens=512,
            ppo_mini_batch_size=16,
        )

        cfg = _ppo_config_from_args(args)

        self.assertEqual(cfg.mini_batch_size, 16)
        self.assertEqual(cfg.logprob_window_strategy, "grouped")
        self.assertEqual(cfg.target_logprob_chunk_tokens, 512)

    def test_next_candidate_adapter_skips_resumed_source_adapter(self) -> None:
        from run_economist_rl_split_workers import SplitWorkerState, next_candidate_adapter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "checkpoints" / "adapters" / "economistRL"
            approved = output_root / "rl_pass_split_001"
            approved.mkdir(parents=True)
            (approved / "adapter_model.safetensors").write_bytes(b"weights")
            state = SplitWorkerState(
                latest_approved_adapter=str(approved),
                latest_adapter_version="rl_pass_split_001",
                ppo_updates=0,
            )

            candidate = next_candidate_adapter(output_root, state)

            self.assertEqual(candidate, (output_root / "rl_pass_split_002").resolve())

    def test_run_descriptor_records_number_date_pipeline_and_extraction_paths(self) -> None:
        from run_economist_rl_split_workers import create_run_descriptor

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_root = root / "benchmarks" / "results" / "economistRL"
            output_root = root / "checkpoints" / "adapters" / "economistRL"
            descriptor = create_run_descriptor(
                results_root=results_root,
                output_root=output_root,
                queue_file=results_root / "split_worker" / "training_queue.jsonl",
                state_file=results_root / "split_worker" / "state.json",
                task_db=root / "benchmarks" / "economistRL_tasks_v3_execution.json",
                init_adapter_path=root / "checkpoints" / "fe-lora-arena-apply-sft",
                entrypoint="scripts/lambda/run_economist_rl_split_workers.py",
                argv=["--max-ppo-updates", "2"],
                started_at_utc="2026-06-13T22:00:00Z",
            )

            self.assertEqual(descriptor["schema_version"], "economist_rl_pipeline_run_v1")
            self.assertEqual(descriptor["run_number"], 1)
            self.assertEqual(descriptor["run_date_utc"], "2026-06-13")
            self.assertEqual(descriptor["run_id"], "run_0001_20260613T220000Z")
            self.assertEqual(descriptor["status"], "running")
            self.assertEqual(descriptor["pipeline_lane"], "economistRL_execution_rl")
            self.assertIn("PPO", descriptor["success_definition"])
            self.assertIn("scripts/adapters/build_economist_rl_dataset.py", descriptor["obsolete_paths"])
            self.assertEqual(descriptor["artifact_roots"]["run_dir"], str(results_root / "runs" / "run_0001_20260613T220000Z"))
            self.assertEqual(descriptor["extraction"]["extract_index"], str(results_root / "extracts" / "index.jsonl"))
            self.assertEqual(descriptor["usable_outputs"]["ppo_candidates"], str(output_root / "rl_pass_*"))

    def test_finalize_run_descriptor_classifies_ppo_success_and_writes_manifest(self) -> None:
        from run_economist_rl_split_workers import create_run_descriptor, finalize_run_descriptor

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_root = root / "benchmarks" / "results" / "economistRL"
            descriptor = create_run_descriptor(
                results_root=results_root,
                output_root=root / "checkpoints" / "adapters" / "economistRL",
                queue_file=results_root / "split_worker" / "training_queue.jsonl",
                state_file=results_root / "split_worker" / "state.json",
                task_db=root / "benchmarks" / "economistRL_tasks_v3_execution.json",
                init_adapter_path=root / "checkpoints" / "fe-lora-arena-apply-sft",
                entrypoint="scripts/lambda/run_economist_rl_split_workers.py",
                argv=[],
                started_at_utc="2026-06-13T22:00:00Z",
            )
            final = finalize_run_descriptor(
                descriptor,
                state={"ppo_updates": 1, "rollout_batches": 3, "last_train_status": "trained"},
                phases=[
                    {"phase": "rollout", "status": "completed", "cycle_id": 7},
                    {"phase": "ppo", "status": "trained", "candidate_adapter": "/tmp/rl_pass_split_001"},
                ],
                finished_at_utc="2026-06-13T22:30:00Z",
            )

            self.assertEqual(final["status"], "completed_trained")
            self.assertEqual(final["finished_at_utc"], "2026-06-13T22:30:00Z")
            manifest_path = Path(final["manifest_file"])
            run_md = Path(final["run_md"])
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(run_md.is_file())
            persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["status"], "completed_trained")
            self.assertIn("Run 0001", run_md.read_text(encoding="utf-8"))

    def test_finalize_run_descriptor_classifies_ppo_failure(self) -> None:
        from run_economist_rl_split_workers import create_run_descriptor, finalize_run_descriptor

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_root = root / "benchmarks" / "results" / "economistRL"
            descriptor = create_run_descriptor(
                results_root=results_root,
                output_root=root / "checkpoints" / "adapters" / "economistRL",
                queue_file=results_root / "split_worker" / "training_queue.jsonl",
                state_file=results_root / "split_worker" / "state.json",
                task_db=root / "benchmarks" / "economistRL_tasks_v3_execution.json",
                init_adapter_path=root / "checkpoints" / "fe-lora-arena-apply-sft",
                entrypoint="scripts/lambda/run_economist_rl_split_workers.py",
                argv=[],
                started_at_utc="2026-06-13T22:00:00Z",
            )

            final = finalize_run_descriptor(
                descriptor,
                state={"ppo_updates": 0, "rollout_batches": 1, "last_train_status": "failed"},
                phases=[{"phase": "ppo", "status": "failed_exception", "error": "subprocess"}],
                finished_at_utc="2026-06-13T22:30:00Z",
            )

            self.assertEqual(final["status"], "failed_ppo")


if __name__ == "__main__":
    unittest.main()
