#!/usr/bin/env python3
"""V1 adapter task-family taxonomy and registry helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = REPO / "training" / "adapter_registry_v1.json"
TAXONOMY_VERSION = "adapter_taxonomy_v1"
ROUTER_POLICY_VERSION = "router_policy_v1"
COUNCIL_THRESHOLDS_VERSION = "council_thresholds_v1"

LOCKED_ADAPTER_FAMILIES: List[str] = [
    "general_fallback",
    "documentation",
    "loading_screen",
    "hud_status",
    "economy_tooltip",
    "combat_risk",
    "save_load_api_guard",
    "ai_planning_explanation",
]

FIRST_WAVE_SPECIALIZED: List[str] = [
    "hud_status",
    "economy_tooltip",
    "combat_risk",
    "save_load_api_guard",
    "ai_planning_explanation",
]

TASK_TO_ADAPTER: Dict[str, str] = {
    "mlx-lora-docs-normalize": "documentation",
    "loading-screen-polish": "loading_screen",
    "hud-status-summary": "hud_status",
    "economy-tooltip": "economy_tooltip",
    "combat-risk-preview": "combat_risk",
    "save-load-api-guard": "save_load_api_guard",
    "ai-planning-explanation": "ai_planning_explanation",
}


@dataclass
class CanaryPolicy:
    traffic_steps: List[int] = field(default_factory=lambda: [5, 15, 35, 60, 100])
    rollback_apply_fail_pp: float = 8.0
    rollback_latency_p95_regression_pct: float = 40.0
    rollback_safety_events: int = 2
    windows_to_promote: int = 2
    hold_on_positive_gain: bool = True


@dataclass
class AdapterEntry:
    adapter_id: str
    task_family: str
    task_ids: List[str]
    base_model: str
    adapter_path: str
    lineage: str
    promotion_state: str = "shadow"
    is_specialized: bool = True
    policy_version: str = ROUTER_POLICY_VERSION
    council_thresholds_version: str = COUNCIL_THRESHOLDS_VERSION
    progressive_context_default: str = "auto"
    canary_policy: CanaryPolicy = field(default_factory=CanaryPolicy)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        data = asdict(self)
        data["canary_policy"] = asdict(self.canary_policy)
        return data


def default_registry(
    *,
    base_model: str = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    checkpoints_root: str = "checkpoints/adapters",
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    entries: List[AdapterEntry] = []
    for family in LOCKED_ADAPTER_FAMILIES:
        is_specialized = family != "general_fallback"
        entries.append(
            AdapterEntry(
                adapter_id=family,
                task_family=family,
                task_ids=[tid for tid, aid in TASK_TO_ADAPTER.items() if aid == family],
                base_model=base_model,
                adapter_path=f"{checkpoints_root}/{family}/champion",
                lineage=f"{family}:v1:champion",
                promotion_state="champion" if family == "general_fallback" else "shadow",
                is_specialized=is_specialized,
            )
        )
    return {
        "schema_version": "adapter_registry_v1",
        "taxonomy_version": TAXONOMY_VERSION,
        "policy_version": ROUTER_POLICY_VERSION,
        "created_at": now,
        "updated_at": now,
        "rollout_profile": "peak_specialist_v1",
        "specialist_mix": {"family_specific": 0.85, "shared_anchor": 0.10, "hard_negative": 0.05},
        "entries": [entry.to_json() for entry in entries],
    }


def write_default_registry(path: Path = DEFAULT_REGISTRY_PATH) -> Path:
    doc = default_registry()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def load_registry(path: Path = DEFAULT_REGISTRY_PATH) -> Dict[str, Any]:
    if not path.is_file():
        return default_registry()
    return json.loads(path.read_text(encoding="utf-8"))


def adapter_for_task(task_id: str, registry: Optional[Dict[str, Any]] = None) -> str:
    if task_id in TASK_TO_ADAPTER:
        return TASK_TO_ADAPTER[task_id]
    reg = registry or default_registry()
    for entry in reg.get("entries", []):
        if task_id in (entry.get("task_ids") or []):
            return str(entry.get("adapter_id") or "general_fallback")
    return "general_fallback"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Write or inspect adapter registry v1")
    ap.add_argument("--write-default", action="store_true", help="Write a default registry file.")
    ap.add_argument("--path", type=Path, default=DEFAULT_REGISTRY_PATH, help="Registry JSON path.")
    args = ap.parse_args()

    if args.write_default:
        out = write_default_registry(args.path)
        print(f"Wrote {out}")
        return
    print(json.dumps(load_registry(args.path), indent=2))


if __name__ == "__main__":
    main()

