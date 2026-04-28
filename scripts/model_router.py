#!/usr/bin/env python3
"""
Cost-aware routing between a local MLX model and an OpenAI-compatible API.

The initial policy is deterministic and auditable: it routes low-risk requests
local, high-risk/broad tasks to frontier, and review/planning tasks to hybrid.
Actual frontier calls require `FRONTIER_API_KEY`; dry-run routing never calls a
paid API.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_LOCAL_MODEL = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"


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

    def __init__(
        self,
        frontier_input_cost_per_million: float = 5.0,
        frontier_output_cost_per_million: float = 15.0,
        long_prompt_tokens: int = 3500,
    ) -> None:
        self.frontier_input_cost_per_million = frontier_input_cost_per_million
        self.frontier_output_cost_per_million = frontier_output_cost_per_million
        self.long_prompt_tokens = long_prompt_tokens

    def decide(self, request: GenerationRequest) -> RouteDecision:
        prompt = request.prompt_text.lower()
        input_tokens = estimate_tokens(request.prompt_text)
        output_tokens = request.max_tokens

        if request.force_route in {"local", "frontier", "hybrid"}:
            route = request.force_route
            reason = f"forced route: {route}"
        elif any(k in prompt for k in self.frontier_keywords):
            route = "frontier"
            reason = "high-risk or broad reasoning keyword matched"
        elif input_tokens >= self.long_prompt_tokens:
            route = "frontier"
            reason = "prompt is too long for cheap local-first routing"
        elif any(k in prompt for k in self.local_keywords):
            route = "local"
            reason = "low-risk local keyword matched"
        elif any(k in prompt for k in self.hybrid_keywords):
            route = "hybrid"
            reason = "review/planning/optimization task benefits from local draft plus frontier review"
        else:
            route = "local"
            reason = "default local route for routine request"

        cost = 0.0
        if route in {"frontier", "hybrid"}:
            cost = (
                input_tokens * self.frontier_input_cost_per_million
                + output_tokens * self.frontier_output_cost_per_million
            ) / 1_000_000
        return RouteDecision(route, reason, input_tokens, output_tokens, cost)


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
