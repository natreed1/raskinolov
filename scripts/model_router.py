#!/usr/bin/env python3
"""
Cost-aware routing between a local MLX model and an OpenAI-compatible API.

The initial policy is deterministic and auditable: it routes low-risk requests
local, high-risk/broad tasks to frontier, and review/planning tasks to hybrid.
Actual frontier calls require `FRONTIER_API_KEY`; dry-run routing never calls a
paid API.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from router.classifier import LinearRouterClassifier
from router.policy import derive_route_from_adapter

DEFAULT_LOCAL_MODEL = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_./*-]+")


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class GenerationRequest:
    messages: List[ChatMessage]
    max_tokens: int = 1024
    temperature: float = 0.0
    force_route: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_text(self) -> str:
        return "\n".join(m.content for m in self.messages)


@dataclass
class RouteDecision:
    route: str
    reason: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    adapter_id: str = "general_fallback"
    confidence: float = 0.5
    ambiguity: float = 0.5
    risk_class: str = "low"
    complexity: str = "low"
    policy_version: str = "router_policy_v2_adapter_first"


@dataclass
class GenerationResult:
    text: str
    route: str
    backend: str
    reason: str
    latency_s: float
    estimated_cost_usd: float
    usage: Dict[str, Any] = field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    # Cheap, provider-agnostic approximation. Good enough for routing/cost logs.
    return max(1, len(text) // 4)


class RoutingPolicy:
    frontier_keywords = {
        "security",
        "auth",
        "authentication",
        "authorization",
        "cryptography",
        "payment",
        "billing",
        "privacy",
        "data loss",
        "incident",
        "production",
        "distributed",
        "race condition",
        "concurrency",
        "architecture",
        "migration",
        "large refactor",
        "multi-service",
        "cross-system",
        "swe-bench",
        "novel algorithm",
    }
    hybrid_keywords = {
        "review",
        "audit",
        "compare",
        "tradeoff",
        "plan",
        "benchmark",
        "performance",
        "optimize",
        "root cause",
    }
    local_keywords = {
        "rename",
        "format",
        "comment",
        "docs",
        "readme",
        "small",
        "simple",
        "boilerplate",
        "explain",
        "summarize",
    }
    specialist_keywords = {
        "loading_screen": (
            "loading screen",
            "load screen",
            "splash screen",
            "title screen",
            "royal",
            "medieval",
            "medival",
            "kingdom",
            "crest",
            "banner",
        ),
        "hud_status": (
            "hud",
            "status panel",
            "status overlay",
            "ui overlay",
            "status summary",
        ),
        "economy_tooltip": (
            "economy",
            "tooltip",
            "income",
            "upkeep",
            "resource",
            "net income",
            "gold per turn",
        ),
        "combat_risk": (
            "combat risk",
            "combat preview",
            "combat previewing",
            "battle risk",
            "risk preview",
            "enemy stats",
            "enemy stats versus",
            "enemy stats vs",
            "versus your own",
            "vs your own",
            "engagement risk",
            "threat",
        ),
        "save_load_api_guard": (
            "save/load",
            "save load",
            "save",
            "autosave",
            "savegame",
            "load game",
            "api guard",
            "auth guard",
            "serialization",
        ),
        "ai_planning_explanation": (
            "ai planning",
            "planning explanation",
            "why ai",
            "decision rationale",
            "explain ai plan",
            "plan",
            "planning",
            "review",
            "tradeoff",
            "optimize",
            "architecture",
        ),
        "documentation": (
            "project_state",
            "session_log",
            "run_history",
            "workflow",
            "documentation",
            "docs/",
            "docs",
            "run log",
            "benchmark script",
            "summarize",
        ),
    }

    def __init__(
        self,
        frontier_input_cost_per_million: float = 5.0,
        frontier_output_cost_per_million: float = 15.0,
        long_prompt_tokens: int = 3500,
        adapter_registry_path: Optional[Path] = None,
        classifier_dir: Optional[Path] = None,
        classifier_confidence_threshold: Optional[float] = None,
    ) -> None:
        self.frontier_input_cost_per_million = frontier_input_cost_per_million
        self.frontier_output_cost_per_million = frontier_output_cost_per_million
        self.long_prompt_tokens = long_prompt_tokens
        default_registry = Path(__file__).resolve().parent.parent / "training" / "adapter_registry_v1.json"
        self.adapter_registry_path = adapter_registry_path or default_registry
        self.available_adapters = self._load_available_adapters()
        env_threshold = os.environ.get("ROUTER_CLASSIFIER_CONFIDENCE_THRESHOLD")
        if classifier_confidence_threshold is None:
            if env_threshold:
                try:
                    classifier_confidence_threshold = float(env_threshold)
                except ValueError:
                    classifier_confidence_threshold = 0.55
            else:
                classifier_confidence_threshold = 0.55
        self.classifier_confidence_threshold = classifier_confidence_threshold
        self.adapter_selection_mode = (
            os.environ.get("ROUTER_ADAPTER_SELECTION_MODE", "hybrid").strip().lower()
        )
        self.similarity_min_score = float(os.environ.get("ROUTER_SIMILARITY_MIN_SCORE", "0.10"))
        self.similarity_min_margin = float(os.environ.get("ROUTER_SIMILARITY_MIN_MARGIN", "0.03"))
        self.unknown_review_enabled = os.environ.get("ROUTER_UNKNOWN_REVIEW_ENABLED", "1").strip() not in {
            "0",
            "false",
            "False",
        }
        self.unknown_review_conf_threshold = float(
            os.environ.get("ROUTER_UNKNOWN_REVIEW_CONFIDENCE_THRESHOLD", "0.62")
        )
        self.unknown_review_jsonl = Path(
            os.environ.get(
                "ROUTER_UNKNOWN_REVIEW_JSONL",
                str(Path(__file__).resolve().parent.parent / "data" / "routing" / "unknown_prompts_review_queue.jsonl"),
            )
        ).expanduser()
        default_classifier_dir = Path(__file__).resolve().parent.parent / "training" / "router_classifier_v1"
        env_classifier_dir = os.environ.get("ROUTER_CLASSIFIER_DIR")
        self.classifier_dir = classifier_dir or Path(env_classifier_dir).expanduser().resolve() if env_classifier_dir else default_classifier_dir
        self.classifier = self._load_classifier()
        self.similarity_prototypes = self._build_similarity_prototypes()

    def _load_available_adapters(self) -> set[str]:
        try:
            payload = json.loads(self.adapter_registry_path.read_text(encoding="utf-8"))
        except Exception:
            return {"general_fallback"}
        rows = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return {"general_fallback"}
        out = {"general_fallback"}
        for row in rows:
            aid = str((row or {}).get("adapter_id") or "").strip()
            if aid:
                out.add(aid)
        return out

    def _load_classifier(self) -> Optional[LinearRouterClassifier]:
        try:
            if not self.classifier_dir.is_dir():
                return None
            return LinearRouterClassifier.load(self.classifier_dir)
        except Exception:
            return None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]

    def _build_similarity_prototypes(self) -> Dict[str, Dict[str, float]]:
        """Build per-adapter lexical prototypes from benchmark task prompts."""
        root = Path(__file__).resolve().parent.parent
        candidates = [
            root / "benchmarks" / "specialist_benchmark_tasks.json",
            root / "benchmarks" / "task_routing_tasks.json",
            root / "benchmarks" / "task_routing_mixed_tasks_v1.json",
            root / "benchmarks" / "documentation_testing_agent_eval_tasks_v1.json",
            root / "benchmarks" / "loading_screen_mass_tasks_v1.json",
            root / "benchmarks" / "hud_status_mass_tasks_v1.json",
            root / "benchmarks" / "economy_tooltip_mass_tasks_v1.json",
            root / "benchmarks" / "combat_risk_mass_tasks_v1.json",
            root / "benchmarks" / "save_load_api_guard_mass_tasks_v1.json",
            root / "benchmarks" / "ai_planning_explanation_mass_tasks_v1.json",
        ]
        per_adapter_docs: Dict[str, List[Dict[str, float]]] = {}
        for path in candidates:
            if not path.is_file():
                continue
            try:
                rows = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                prompt = str(row.get("prompt") or "")
                adapters_raw = row.get("specialists")
                if isinstance(adapters_raw, list):
                    adapters = [str(a).strip() for a in adapters_raw if str(a).strip()]
                else:
                    expected = str(row.get("expected_adapter_id") or "").strip()
                    adapters = [expected] if expected else []
                if not prompt or not adapters:
                    continue
                toks = self._tokenize(prompt)
                if not toks:
                    continue
                vec: Dict[str, float] = {}
                for tok in toks:
                    vec[tok] = vec.get(tok, 0.0) + 1.0
                # Sublinear TF to reduce long prompt dominance.
                for tok in list(vec.keys()):
                    vec[tok] = float(math.log1p(vec[tok])) if vec[tok] > 0 else 0.0
                for adapter_id in adapters:
                    aid = str(adapter_id).strip()
                    if not aid or aid not in self.available_adapters:
                        continue
                    per_adapter_docs.setdefault(aid, []).append(vec)
        prototypes: Dict[str, Dict[str, float]] = {}
        for aid, docs in per_adapter_docs.items():
            merged: Dict[str, float] = {}
            if not docs:
                continue
            for d in docs:
                for tok, val in d.items():
                    merged[tok] = merged.get(tok, 0.0) + float(val)
            n = float(len(docs))
            for tok in list(merged.keys()):
                merged[tok] /= n
            prototypes[aid] = merged
        return prototypes

    @staticmethod
    def _cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = 0.0
        for tok, av in a.items():
            bv = b.get(tok)
            if bv is not None:
                dot += av * bv
        na = sum(v * v for v in a.values()) ** 0.5
        nb = sum(v * v for v in b.values()) ** 0.5
        if na <= 1e-12 or nb <= 1e-12:
            return 0.0
        return dot / (na * nb)

    def _classify_adapter_similarity(self, prompt: str) -> tuple[str, float, str]:
        if not self.similarity_prototypes:
            return "general_fallback", 0.45, "no similarity prototypes available"
        toks = self._tokenize(prompt)
        if not toks:
            return "general_fallback", 0.45, "empty prompt tokens for similarity routing"
        query: Dict[str, float] = {}
        for tok in toks:
            query[tok] = query.get(tok, 0.0) + 1.0
        for tok in list(query.keys()):
            query[tok] = float(math.log1p(query[tok])) if query[tok] > 0 else 0.0
        scores: List[tuple[float, str]] = []
        for aid, proto in self.similarity_prototypes.items():
            score = self._cosine_sparse(query, proto)
            scores.append((score, aid))
        scores.sort(reverse=True)
        best_score, best_adapter = scores[0]
        second = scores[1][0] if len(scores) > 1 else 0.0
        margin = best_score - second
        conf = max(0.5, min(0.98, 0.55 + 0.6 * best_score - 0.25 * second))
        if best_score < self.similarity_min_score:
            return (
                "general_fallback",
                0.5,
                (
                    "low cosine similarity across adapter prototypes "
                    f"(sim={best_score:.3f}, threshold={self.similarity_min_score:.3f})"
                ),
            )
        if margin < self.similarity_min_margin:
            return (
                "general_fallback",
                0.5,
                (
                    "ambiguous cosine match (near tie) "
                    f"(sim={best_score:.3f}, second={second:.3f}, margin={margin:.3f}, "
                    f"min_margin={self.similarity_min_margin:.3f})"
                ),
            )
        return (
            best_adapter,
            conf,
            f"cosine prototype match (sim={best_score:.3f}, second={second:.3f}, margin={margin:.3f})",
        )

    @staticmethod
    def _contains_keyword(prompt: str, keyword: str) -> bool:
        escaped = re.escape(keyword).replace(r"\ ", r"\s+")
        pattern = rf"(?<![a-z0-9_]){escaped}(?![a-z0-9_])"
        return re.search(pattern, prompt) is not None

    @classmethod
    def _signal_count(cls, prompt: str, keywords: Iterable[str]) -> int:
        return sum(1 for kw in keywords if cls._contains_keyword(prompt, kw))

    @classmethod
    def _matches_any(cls, prompt: str, keywords: Iterable[str]) -> bool:
        return any(cls._contains_keyword(prompt, kw) for kw in keywords)

    def _classify_adapter_keywords(self, prompt: str) -> tuple[str, float, str]:
        if (
            self._contains_keyword(prompt, "loading screen")
            or self._contains_keyword(prompt, "load screen")
            or self._contains_keyword(prompt, "splash screen")
        ):
            if "loading_screen" in self.available_adapters:
                return "loading_screen", 0.92, "explicit loading-screen phrase match"

        ranked: list[tuple[int, str]] = []
        for adapter_id, keywords in self.specialist_keywords.items():
            if adapter_id not in self.available_adapters:
                continue
            hits = self._signal_count(prompt, keywords)
            if hits > 0:
                ranked.append((hits, adapter_id))
        if not ranked:
            return "general_fallback", 0.45, "no specialist keyword match"
        ranked.sort(reverse=True)
        top_hits, top_adapter = ranked[0]
        second_hits = ranked[1][0] if len(ranked) > 1 else 0
        confidence = min(0.98, 0.55 + 0.12 * top_hits - 0.06 * second_hits)
        return top_adapter, max(0.5, confidence), f"specialist keyword match ({top_hits} hits)"

    def _classify_adapter(self, prompt: str) -> tuple[str, float, str]:
        if self.adapter_selection_mode == "similarity":
            return self._classify_adapter_similarity(prompt)
        if self.adapter_selection_mode == "lexical":
            return self._classify_adapter_keywords(prompt)
        if self.classifier is not None:
            pred = self.classifier.predict(prompt)
            classifier_adapter = pred.adapter_id if pred.adapter_id in self.available_adapters else "general_fallback"
            if pred.confidence >= self.classifier_confidence_threshold:
                return (
                    classifier_adapter,
                    pred.confidence,
                    f"oss linear classifier match (p={pred.confidence:.2f})",
                )
            lexical_adapter, lexical_conf, lexical_reason = self._classify_adapter_keywords(prompt)
            return (
                lexical_adapter,
                max(lexical_conf, pred.confidence * 0.95),
                (
                    f"classifier low confidence (p={pred.confidence:.2f}) "
                    f"-> lexical fallback: {lexical_reason}"
                ),
            )
        return self._classify_adapter_keywords(prompt)

    def _enqueue_unknown_prompt(
        self,
        *,
        prompt_text: str,
        route: str,
        adapter_id: str,
        confidence: float,
        ambiguity: float,
        reason: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        path = self.unknown_review_jsonl.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        prompt_norm = (prompt_text or "").strip()
        payload = {
            "schema_version": "router_unknown_prompt_review_v1",
            "captured_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "prompt": prompt_norm,
            "prompt_sha256": hashlib.sha256(prompt_norm.encode("utf-8")).hexdigest() if prompt_norm else "",
            "predicted_adapter_id": adapter_id,
            "predicted_route": route,
            "confidence": round(float(confidence), 4),
            "ambiguity": round(float(ambiguity), 4),
            "input_tokens_est": int(input_tokens),
            "output_tokens_est": int(output_tokens),
            "selection_mode": self.adapter_selection_mode,
            "reason": reason,
            "review_status": "pending",
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def decide(self, request: GenerationRequest) -> RouteDecision:
        prompt = request.prompt_text.lower()
        input_tokens = estimate_tokens(request.prompt_text)
        output_tokens = request.max_tokens
        adapter_id, confidence, adapter_reason = self._classify_adapter(prompt)

        frontier_matched = self._matches_any(prompt, self.frontier_keywords)
        hybrid_matched = self._matches_any(prompt, self.hybrid_keywords)
        route, reason = derive_route_from_adapter(
            adapter_id=adapter_id,
            input_tokens=input_tokens,
            force_route=request.force_route,
            long_prompt_tokens=self.long_prompt_tokens,
            frontier_keywords_matched=frontier_matched,
            hybrid_keywords_matched=hybrid_matched,
        )

        if frontier_matched:
            risk_class = "high"
        elif hybrid_matched:
            risk_class = "medium"
        else:
            risk_class = "low"

        if input_tokens >= self.long_prompt_tokens:
            complexity = "high"
        elif input_tokens >= 1200:
            complexity = "medium"
        else:
            complexity = "low"

        cost = 0.0
        if route in {"frontier", "hybrid"}:
            cost = (
                input_tokens * self.frontier_input_cost_per_million
                + output_tokens * self.frontier_output_cost_per_million
            ) / 1_000_000
        merged_reason = f"{reason}; {adapter_reason}; adapter={adapter_id}"
        ambiguity = max(0.0, 1.0 - confidence)

        should_queue_unknown = (
            self.unknown_review_enabled
            and (
                adapter_id == "general_fallback"
                or confidence < self.unknown_review_conf_threshold
                or "low cosine similarity" in adapter_reason.lower()
            )
        )
        if should_queue_unknown:
            try:
                self._enqueue_unknown_prompt(
                    prompt_text=request.prompt_text,
                    route=route,
                    adapter_id=adapter_id,
                    confidence=confidence,
                    ambiguity=ambiguity,
                    reason=merged_reason,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                merged_reason = f"{merged_reason}; unknown_review=queued"
            except OSError:
                merged_reason = f"{merged_reason}; unknown_review=queue_failed"

        return RouteDecision(
            route=route,
            reason=merged_reason,
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            estimated_cost_usd=cost,
            adapter_id=adapter_id,
            confidence=confidence,
            ambiguity=ambiguity,
            risk_class=risk_class,
            complexity=complexity,
            policy_version="router_policy_v2_adapter_first",
        )


class LocalMlxBackend:
    def __init__(
        self,
        model_id: str = DEFAULT_LOCAL_MODEL,
        adapter_path: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.adapter_path = adapter_path
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler

        self._generate = generate
        self._make_sampler = make_sampler
        load_kw: Dict[str, Any] = {}
        if self.adapter_path:
            load_kw["adapter_path"] = self.adapter_path
        self._model, self._tokenizer = load(self.model_id, **load_kw)
        return self._model, self._tokenizer

    def generate(self, request: GenerationRequest) -> str:
        model, tokenizer = self._ensure_loaded()
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        kwargs: Dict[str, Any] = {"max_tokens": request.max_tokens}
        if request.temperature > 0:
            kwargs["sampler"] = self._make_sampler(temp=request.temperature, top_p=1.0)
        return self._generate(model, tokenizer, prompt=prompt, verbose=False, **kwargs)


class OpenAICompatibleBackend:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: int = 120,
    ) -> None:
        self.base_url = (base_url or os.environ.get("FRONTIER_API_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("FRONTIER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("FRONTIER_MODEL") or "gpt-4o-mini"
        self.timeout_s = timeout_s

    def generate(self, request: GenerationRequest) -> tuple[str, Dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError("FRONTIER_API_KEY or OPENAI_API_KEY is required for frontier calls.")
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        body = self._post_chat_completion(payload)
        text = body["choices"][0]["message"]["content"]
        return text, body.get("usage", {})

    def _post_chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_text = exc.read().decode("utf-8", "replace")
            if exc.code == 400 and "max_tokens" in error_text and "max_completion_tokens" in error_text:
                retry_payload = dict(payload)
                retry_payload["max_completion_tokens"] = retry_payload.pop("max_tokens")
                return self._post_chat_completion(retry_payload)
            if exc.code == 400 and "temperature" in error_text and "Unsupported value" in error_text:
                retry_payload = dict(payload)
                retry_payload.pop("temperature", None)
                return self._post_chat_completion(retry_payload)
            raise RuntimeError(f"Frontier API HTTP {exc.code}: {error_text}") from exc


class ModelRouter:
    def __init__(
        self,
        policy: Optional[RoutingPolicy] = None,
        local_backend: Optional[LocalMlxBackend] = None,
        frontier_backend: Optional[OpenAICompatibleBackend] = None,
    ) -> None:
        self.policy = policy or RoutingPolicy()
        self.local_backend = local_backend or LocalMlxBackend()
        self.frontier_backend = frontier_backend or OpenAICompatibleBackend()

    def decide(self, request: GenerationRequest) -> RouteDecision:
        return self.policy.decide(request)

    def generate(self, request: GenerationRequest, dry_run: bool = False) -> GenerationResult:
        decision = self.decide(request)
        if dry_run:
            return GenerationResult(
                text="",
                route=decision.route,
                backend="dry-run",
                reason=decision.reason,
                latency_s=0.0,
                estimated_cost_usd=decision.estimated_cost_usd,
            )

        t0 = time.perf_counter()
        if decision.route == "local":
            text = self.local_backend.generate(request)
            backend = "local_mlx"
            usage: Dict[str, Any] = {}
        elif decision.route == "frontier":
            text, usage = self.frontier_backend.generate(request)
            backend = "openai_compatible"
        else:
            local_text = self.local_backend.generate(request)
            review_request = GenerationRequest(
                messages=[
                    ChatMessage(
                        "system",
                        "Review and improve the local model draft. Return a concise final answer.",
                    ),
                    ChatMessage("user", request.prompt_text),
                    ChatMessage("assistant", local_text),
                ],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            text, usage = self.frontier_backend.generate(review_request)
            backend = "hybrid_local_then_frontier"

        return GenerationResult(
            text=text,
            route=decision.route,
            backend=backend,
            reason=decision.reason,
            latency_s=time.perf_counter() - t0,
            estimated_cost_usd=decision.estimated_cost_usd,
            usage=usage,
        )


def messages_from_prompt(prompt: str, system_prompt: str = "You are a helpful coding assistant.") -> List[ChatMessage]:
    return [ChatMessage("system", system_prompt), ChatMessage("user", prompt)]


def route_prompt(prompt: str, max_tokens: int = 1024, force_route: Optional[str] = None) -> RouteDecision:
    req = GenerationRequest(messages=messages_from_prompt(prompt), max_tokens=max_tokens, force_route=force_route)
    return RoutingPolicy().decide(req)
