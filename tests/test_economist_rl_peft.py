"""PEFT adapter layout helpers for economistRL Transformers paths."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from economist_rl_peft import (  # noqa: E402
    adapter_dir_has_weights,
    adapter_weights_file,
    build_peft_adapter_config,
    infer_target_modules_from_mlx_keys,
    is_mlx_lora_adapter_dir,
    is_peft_adapter_dir,
    mlx_lora_key_to_peft_key,
)


class EconomistRLPeftHelperTests(unittest.TestCase):
    def test_mlx_key_to_peft_key(self) -> None:
        self.assertEqual(
            mlx_lora_key_to_peft_key("model.layers.0.mlp.down_proj.lora_a"),
            "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight",
        )
        self.assertEqual(
            mlx_lora_key_to_peft_key("model.layers.3.self_attn.q_proj.lora_b"),
            "base_model.model.model.layers.3.self_attn.q_proj.lora_B.weight",
        )

    def test_infer_target_modules(self) -> None:
        keys = [
            "model.layers.0.self_attn.q_proj.lora_a",
            "model.layers.0.self_attn.v_proj.lora_a",
            "model.layers.1.mlp.down_proj.lora_a",
        ]
        self.assertEqual(
            infer_target_modules_from_mlx_keys(keys),
            ["down_proj", "q_proj", "v_proj"],
        )

    def test_peft_adapter_dir_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "adapter_model.safetensors").write_bytes(b"")
            (root / "adapter_config.json").write_text(
                json.dumps(build_peft_adapter_config(
                    base_model_name_or_path="Qwen/Qwen2.5-Coder-7B-Instruct",
                    target_modules=["q_proj"],
                    rank=16,
                    lora_alpha=20.0,
                    lora_dropout=0.05,
                )),
                encoding="utf-8",
            )
            self.assertTrue(is_peft_adapter_dir(root))
            self.assertFalse(is_mlx_lora_adapter_dir(root))

    def test_mlx_adapter_dir_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "adapters.safetensors").write_bytes(b"")
            (root / "adapter_config.json").write_text(
                json.dumps({"fine_tune_type": "lora", "lora_parameters": {"rank": 16, "scale": 20.0}}),
                encoding="utf-8",
            )
            self.assertTrue(is_mlx_lora_adapter_dir(root))
            self.assertFalse(is_peft_adapter_dir(root))
            self.assertTrue(adapter_dir_has_weights(root))
            self.assertEqual(adapter_weights_file(root).name, "adapters.safetensors")

    def test_peft_weights_detected_for_chaining(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "adapter_model.safetensors").write_bytes(b"")
            (root / "adapter_config.json").write_text(
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
            self.assertTrue(adapter_dir_has_weights(root))
            self.assertFalse((root / "adapters.safetensors").is_file())
            self.assertEqual(adapter_weights_file(root).name, "adapter_model.safetensors")


if __name__ == "__main__":
    unittest.main()
