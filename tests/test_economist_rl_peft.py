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
    load_peft_causal_lm,
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

    def test_trainable_loader_uses_4bit_quantization_when_requested(self) -> None:
        from types import ModuleType
        from unittest.mock import patch

        calls: dict[str, object] = {}

        class FakeBitsAndBytesConfig:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        class FakeTokenizer:
            @staticmethod
            def from_pretrained(model_id: str):
                calls["tokenizer_model_id"] = model_id
                return object()

        class FakeBase:
            def __init__(self) -> None:
                self.to_calls: list[object] = []

            def to(self, device):
                self.to_calls.append(device)
                return self

            def train(self):
                calls["train_called"] = True
                return self

            def eval(self):
                calls["eval_called"] = True
                return self

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(model_id: str, **kwargs):
                base = FakeBase()
                calls["model_id"] = model_id
                calls["model_kwargs"] = kwargs
                calls["base"] = base
                return base

        class FakePeftModel:
            @staticmethod
            def from_pretrained(base, adapter_path: str, **kwargs):
                calls["peft_base"] = base
                calls["peft_adapter_path"] = adapter_path
                calls["peft_kwargs"] = kwargs
                return base

        def fake_prepare_model_for_kbit_training(base):
            calls["prepared_for_kbit"] = True
            return base

        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoModelForCausalLM = FakeAutoModel
        fake_transformers.AutoTokenizer = FakeTokenizer
        fake_transformers.BitsAndBytesConfig = FakeBitsAndBytesConfig

        fake_peft = ModuleType("peft")
        fake_peft.PeftModel = FakePeftModel
        fake_peft.prepare_model_for_kbit_training = fake_prepare_model_for_kbit_training

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

            with patch.dict(sys.modules, {"transformers": fake_transformers, "peft": fake_peft}):
                load_peft_causal_lm(
                    "Qwen/Qwen2.5-Coder-7B-Instruct",
                    root,
                    device="cuda",
                    dtype="float16",
                    trainable=True,
                    load_in_4bit=True,
                )

        model_kwargs = calls["model_kwargs"]
        self.assertIsInstance(model_kwargs, dict)
        quant_config = model_kwargs.get("quantization_config")
        self.assertIsInstance(quant_config, FakeBitsAndBytesConfig)
        self.assertEqual(
            quant_config.kwargs,
            {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            },
        )
        self.assertEqual(model_kwargs.get("device_map"), {"": "cuda"})
        self.assertNotIn("torch_dtype", model_kwargs)
        self.assertEqual(calls.get("prepared_for_kbit"), True)
        self.assertEqual(calls["peft_kwargs"], {"is_trainable": True})
        self.assertEqual(calls["base"].to_calls, [])

    def test_loader_can_require_local_hf_cache_without_network_lookups(self) -> None:
        from types import ModuleType
        from unittest.mock import patch

        calls: dict[str, object] = {}

        class FakeTokenizer:
            @staticmethod
            def from_pretrained(model_id: str, **kwargs):
                calls["tokenizer_model_id"] = model_id
                calls["tokenizer_kwargs"] = kwargs
                return object()

        class FakeBase:
            def __init__(self) -> None:
                self.to_calls: list[object] = []

            def to(self, device):
                self.to_calls.append(device)
                return self

            def eval(self):
                calls["eval_called"] = True
                return self

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(model_id: str, **kwargs):
                base = FakeBase()
                calls["model_id"] = model_id
                calls["model_kwargs"] = kwargs
                return base

        class FakePeftModel:
            @staticmethod
            def from_pretrained(base, adapter_path: str, **kwargs):
                calls["peft_kwargs"] = kwargs
                return base

        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoModelForCausalLM = FakeAutoModel
        fake_transformers.AutoTokenizer = FakeTokenizer

        fake_peft = ModuleType("peft")
        fake_peft.PeftModel = FakePeftModel

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

            with patch.dict(sys.modules, {"transformers": fake_transformers, "peft": fake_peft}):
                load_peft_causal_lm(
                    "Qwen/Qwen2.5-Coder-7B-Instruct",
                    root,
                    device="cpu",
                    dtype="float32",
                    require_local_files=True,
                )

        self.assertEqual(calls["tokenizer_kwargs"], {"local_files_only": True})
        self.assertEqual(calls["model_kwargs"], {"torch_dtype": "float32", "local_files_only": True})
        self.assertEqual(calls["peft_kwargs"], {"is_trainable": False, "local_files_only": True})


if __name__ == "__main__":
    unittest.main()
