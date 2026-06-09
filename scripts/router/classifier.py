#!/usr/bin/env python3
"""Open-source adapter classifier (bag-of-words + linear softmax)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

TOKEN_RE = re.compile(r"[a-zA-Z0-9_./*-]+")


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def _feature_matrix(texts: Sequence[str], vocab: dict[str, int]) -> np.ndarray:
    x = np.zeros((len(texts), len(vocab) + 1), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        x[row_idx, 0] = 1.0  # bias feature
        for tok in _tokenize(text):
            idx = vocab.get(tok)
            if idx is None:
                continue
            x[row_idx, idx + 1] += 1.0
    # Sublinear tf scaling keeps long prompts from dominating.
    np.log1p(x[:, 1:], out=x[:, 1:])
    return x


def _build_vocab(texts: Sequence[str], *, max_vocab: int, min_count: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for text in texts:
        for tok in _tokenize(text):
            counts[tok] = counts.get(tok, 0) + 1
    ranked = sorted(
        ((count, tok) for tok, count in counts.items() if count >= min_count),
        reverse=True,
    )
    vocab = {tok: idx for idx, (_, tok) in enumerate(ranked[:max_vocab])}
    return vocab


@dataclass(frozen=True)
class Prediction:
    adapter_id: str
    confidence: float
    ambiguity: float
    probs: Dict[str, float]


class LinearRouterClassifier:
    """Simple OSS linear classifier for adapter-id predictions."""

    def __init__(
        self,
        vocab: dict[str, int],
        labels: list[str],
        weights: np.ndarray,
        *,
        model_version: str = "router_classifier_linear_v1",
    ) -> None:
        self.vocab = vocab
        self.labels = labels
        self.weights = weights.astype(np.float32)
        self.model_version = model_version

    @classmethod
    def load(cls, model_dir: Path) -> "LinearRouterClassifier":
        model_dir = model_dir.expanduser().resolve()
        meta = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
        vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
        labels = list(meta["labels"])
        weights = np.load(model_dir / "weights.npy")
        return cls(
            vocab=vocab,
            labels=labels,
            weights=weights,
            model_version=str(meta.get("model_version") or "router_classifier_linear_v1"),
        )

    def predict(self, text: str) -> Prediction:
        x = _feature_matrix([text], self.vocab)
        probs = _softmax(x @ self.weights.T)[0]
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        adapter_id = self.labels[best_idx]
        dist = {label: float(prob) for label, prob in zip(self.labels, probs)}
        return Prediction(
            adapter_id=adapter_id,
            confidence=confidence,
            ambiguity=max(0.0, 1.0 - confidence),
            probs=dist,
        )


def train_classifier(
    *,
    train_rows: Sequence[dict[str, Any]],
    valid_rows: Sequence[dict[str, Any]],
    max_vocab: int = 8192,
    min_count: int = 1,
    lr: float = 0.2,
    epochs: int = 100,
    l2: float = 5e-4,
    seed: int = 42,
) -> tuple[LinearRouterClassifier, dict[str, Any]]:
    if not train_rows:
        raise ValueError("No train rows provided.")
    texts = [str(row["prompt"]) for row in train_rows]
    labels = sorted({str(row["expected_adapter_id"]) for row in train_rows})
    if not labels:
        raise ValueError("No adapter labels in training data.")
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    y = np.array([label_to_idx[str(row["expected_adapter_id"])] for row in train_rows], dtype=np.int64)
    vocab = _build_vocab(texts, max_vocab=max_vocab, min_count=min_count)
    x = _feature_matrix(texts, vocab)

    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=(len(labels), x.shape[1])).astype(np.float32)

    for _ in range(max(1, epochs)):
        logits = x @ weights.T
        probs = _softmax(logits)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(y)), y] = 1.0
        grad = ((probs - one_hot).T @ x) / float(len(y))
        grad += l2 * weights
        weights -= lr * grad.astype(np.float32)

    clf = LinearRouterClassifier(vocab=vocab, labels=labels, weights=weights)
    train_metrics = evaluate_classifier(clf, train_rows)
    valid_metrics = evaluate_classifier(clf, valid_rows) if valid_rows else {"accuracy": None, "rows": 0}
    summary = {
        "train_accuracy": train_metrics["accuracy"],
        "valid_accuracy": valid_metrics["accuracy"],
        "labels": labels,
        "vocab_size": len(vocab),
        "model_version": clf.model_version,
        "epochs": int(epochs),
        "learning_rate": float(lr),
        "l2": float(l2),
    }
    return clf, summary


def evaluate_classifier(clf: LinearRouterClassifier, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    passed = 0
    per_label: dict[str, dict[str, int]] = {}
    for row in rows:
        expected = str(row["expected_adapter_id"])
        pred = clf.predict(str(row["prompt"]))
        ok = pred.adapter_id == expected
        total += 1
        passed += int(ok)
        stats = per_label.setdefault(expected, {"total": 0, "passed": 0})
        stats["total"] += 1
        stats["passed"] += int(ok)
    accuracy = (passed / total) if total else 0.0
    return {
        "rows": total,
        "passed": passed,
        "accuracy": round(accuracy, 4),
        "per_label": {
            label: {
                "total": stats["total"],
                "passed": stats["passed"],
                "accuracy": round(stats["passed"] / stats["total"], 4) if stats["total"] else 0.0,
            }
            for label, stats in sorted(per_label.items())
        },
    }


def save_classifier(
    clf: LinearRouterClassifier,
    *,
    out_dir: Path,
    train_summary: dict[str, Any],
    dataset_manifest: dict[str, Any],
) -> Path:
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vocab.json").write_text(json.dumps(clf.vocab, indent=2, sort_keys=True), encoding="utf-8")
    np.save(out_dir / "weights.npy", clf.weights)
    manifest = {
        "schema_version": "router_classifier_artifact_v1",
        "model_version": clf.model_version,
        "labels": clf.labels,
        "feature_type": "bow_log_tf_linear_softmax",
        "vocab_size": len(clf.vocab),
        "weights_shape": list(clf.weights.shape),
        "train_summary": train_summary,
        "dataset_manifest": dataset_manifest,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir

