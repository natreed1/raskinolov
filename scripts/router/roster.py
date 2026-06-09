"""Expert roster state machine for council participation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

VALID_STATES = {"candidate", "active", "probation", "demoted"}
VALID_PERSONALITY_STATES = {"candidate", "active", "cooldown", "retired"}
TRAIT_KEYS = (
    "assertiveness",
    "verbosity",
    "risk_tolerance",
    "creativity",
    "skepticism",
    "decisiveness",
)


def _clamp_trait(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _personality_default_state(expert_type: str) -> str:
    return "active" if expert_type == "generalist_profile" else "candidate"


def _default_traits_for(expert_id: str, expert_type: str) -> dict[str, float]:
    if expert_type == "generalist_profile":
        if expert_id == "wide_compressed":
            return {
                "assertiveness": 0.60,
                "verbosity": 0.62,
                "risk_tolerance": 0.55,
                "creativity": 0.52,
                "skepticism": 0.45,
                "decisiveness": 0.62,
            }
        if expert_id == "precise_short":
            return {
                "assertiveness": 0.45,
                "verbosity": 0.40,
                "risk_tolerance": 0.35,
                "creativity": 0.30,
                "skepticism": 0.72,
                "decisiveness": 0.68,
            }
        return {
            "assertiveness": 0.50,
            "verbosity": 0.54,
            "risk_tolerance": 0.45,
            "creativity": 0.42,
            "skepticism": 0.58,
            "decisiveness": 0.56,
        }
    return {
        "assertiveness": 0.60,
        "verbosity": 0.50,
        "risk_tolerance": 0.50,
        "creativity": 0.45,
        "skepticism": 0.60,
        "decisiveness": 0.65,
    }


def _traits_with_overrides(base: dict[str, float], overrides: dict[str, Any] | None) -> dict[str, float]:
    out = {k: _clamp_trait(float(base.get(k, 0.5))) for k in TRAIT_KEYS}
    if isinstance(overrides, dict):
        for key in TRAIT_KEYS:
            if key in overrides and overrides.get(key) is not None:
                try:
                    out[key] = _clamp_trait(float(overrides.get(key)))
                except (TypeError, ValueError):
                    pass
    return out


def _default_personalities_for(
    expert_id: str,
    expert_type: str,
    base_traits: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    base = _traits_with_overrides(
        _default_traits_for(expert_id, expert_type),
        base_traits if isinstance(base_traits, dict) else None,
    )
    if expert_type == "generalist_profile":
        return [{"name": "baseline", "state": "active", "traits": base}]
    return [
        {
            "name": "risk_auditor",
            "state": "candidate",
            "traits": _traits_with_overrides(
                base,
                {
                    "assertiveness": 0.42,
                    "verbosity": 0.44,
                    "risk_tolerance": 0.24,
                    "creativity": 0.35,
                    "skepticism": 0.82,
                    "decisiveness": 0.66,
                },
            ),
        },
        {
            "name": "balanced_operator",
            "state": "candidate",
            "traits": base,
        },
        {
            "name": "direct_builder",
            "state": "candidate",
            "traits": _traits_with_overrides(
                base,
                {
                    "assertiveness": 0.78,
                    "verbosity": 0.55,
                    "risk_tolerance": 0.64,
                    "creativity": 0.56,
                    "skepticism": 0.48,
                    "decisiveness": 0.82,
                },
            ),
        },
    ]


def _normalize_personalities(
    *,
    expert_id: str,
    expert_type: str,
    base_traits: dict[str, float],
    personalities: list[Any] | None,
) -> list[dict[str, Any]]:
    source = personalities if isinstance(personalities, list) and personalities else None
    if source is None:
        source = _default_personalities_for(expert_id, expert_type, base_traits)

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, raw in enumerate(source, start=1):
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or raw.get("id") or f"personality_{idx}").strip()
        if not name:
            name = f"personality_{idx}"
        state = str(raw.get("state") or _personality_default_state(expert_type)).strip()
        if state not in VALID_PERSONALITY_STATES:
            state = _personality_default_state(expert_type)
        traits = _traits_with_overrides(base_traits, raw.get("traits") if isinstance(raw.get("traits"), dict) else None)
        if isinstance(raw.get("trait_deltas"), dict):
            for key in TRAIT_KEYS:
                delta = raw["trait_deltas"].get(key)
                if delta is not None:
                    try:
                        traits[key] = _clamp_trait(float(traits.get(key, 0.5)) + float(delta))
                    except (TypeError, ValueError):
                        pass
        unique_name = name
        suffix = 2
        while unique_name in seen:
            unique_name = f"{name}_{suffix}"
            suffix += 1
        seen.add(unique_name)
        out.append(
            {
                "name": unique_name,
                "state": state,
                "traits": traits,
                "offline_score": _safe_float(raw.get("offline_score"), 0.0),
                "offline_sample_count": int(_safe_float(raw.get("offline_sample_count"), 0.0)),
                "online_task_outcome": _safe_float(raw.get("online_task_outcome"), 0.0),
                "online_sample_count": int(_safe_float(raw.get("online_sample_count"), 0.0)),
            }
        )
    if out:
        return out
    return _default_personalities_for(expert_id, expert_type, base_traits)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_traits(
    *,
    expert_id: str,
    expert_type: str,
    traits: dict[str, Any] | None,
    assertiveness_fallback: float | None = None,
) -> dict[str, float]:
    base = _default_traits_for(expert_id, expert_type)
    if isinstance(traits, dict):
        for key in TRAIT_KEYS:
            if key in traits and traits.get(key) is not None:
                try:
                    base[key] = _clamp_trait(float(traits.get(key)))
                except (TypeError, ValueError):
                    pass
    if assertiveness_fallback is not None:
        base["assertiveness"] = _clamp_trait(assertiveness_fallback)
    return {k: _clamp_trait(float(base[k])) for k in TRAIT_KEYS}


@dataclass
class RosterEntry:
    expert_id: str
    expert_type: str
    state: str
    assertiveness: float = 0.6
    traits: dict[str, float] = field(default_factory=dict)
    personalities: list[dict[str, Any]] = field(default_factory=list)
    offline_score: float = 0.0
    online_task_outcome: float = 0.0
    online_sample_count: int = 0
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        norm_traits = _normalize_traits(
            expert_id=self.expert_id,
            expert_type=self.expert_type,
            traits=self.traits,
            assertiveness_fallback=self.assertiveness,
        )
        return {
            "expert_id": self.expert_id,
            "expert_type": self.expert_type,
            "state": self.state,
            "assertiveness": round(float(norm_traits["assertiveness"]), 4),
            "traits": {k: round(float(v), 4) for k, v in norm_traits.items()},
            "personalities": [
                {
                    "name": str(personality.get("name") or ""),
                    "state": str(personality.get("state") or _personality_default_state(self.expert_type)),
                    "traits": {
                        k: round(float(v), 4)
                        for k, v in sorted(dict(personality.get("traits") or {}).items())
                    },
                    "offline_score": round(float(personality.get("offline_score", 0.0) or 0.0), 4),
                    "offline_sample_count": int(personality.get("offline_sample_count", 0) or 0),
                    "online_task_outcome": round(float(personality.get("online_task_outcome", 0.0) or 0.0), 4),
                    "online_sample_count": int(personality.get("online_sample_count", 0) or 0),
                }
                for personality in _normalize_personalities(
                    expert_id=self.expert_id,
                    expert_type=self.expert_type,
                    base_traits=norm_traits,
                    personalities=self.personalities,
                )
            ],
            "offline_score": round(float(self.offline_score), 4),
            "online_task_outcome": round(float(self.online_task_outcome), 4),
            "online_sample_count": int(self.online_sample_count),
            "updated_at": self.updated_at,
        }


class ExpertRoster:
    def __init__(self, entries: dict[str, RosterEntry]) -> None:
        self.entries = entries

    @classmethod
    def bootstrap(
        cls,
        *,
        specialist_ids: Iterable[str],
    ) -> "ExpertRoster":
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        entries: dict[str, RosterEntry] = {}
        for expert_id in ("wide_compressed", "precise_short", "sliding_window"):
            traits = _default_traits_for(expert_id, "generalist_profile")
            entries[expert_id] = RosterEntry(
                expert_id=expert_id,
                expert_type="generalist_profile",
                state="active",
                assertiveness=traits["assertiveness"],
                traits=traits,
                personalities=_default_personalities_for(expert_id, "generalist_profile"),
                updated_at=now,
            )
        for expert_id in sorted(set(str(x).strip() for x in specialist_ids if str(x).strip())):
            if expert_id == "general_fallback":
                continue
            traits = _default_traits_for(expert_id, "specialist_adapter")
            entries[expert_id] = RosterEntry(
                expert_id=expert_id,
                expert_type="specialist_adapter",
                state="candidate",
                assertiveness=traits["assertiveness"],
                traits=traits,
                personalities=_default_personalities_for(expert_id, "specialist_adapter"),
                updated_at=now,
            )
        return cls(entries)

    @classmethod
    def load_or_bootstrap(
        cls,
        *,
        path: Path,
        specialist_ids: Iterable[str],
    ) -> "ExpertRoster":
        if not path.is_file():
            return cls.bootstrap(specialist_ids=specialist_ids)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return cls.bootstrap(specialist_ids=specialist_ids)
        entries: dict[str, RosterEntry] = {}
        for row in rows:
            expert_id = str((row or {}).get("expert_id") or "").strip()
            if not expert_id:
                continue
            state = str((row or {}).get("state") or "candidate").strip()
            if state not in VALID_STATES:
                state = "candidate"
            expert_type = str((row or {}).get("expert_type") or "specialist_adapter")
            traits = _normalize_traits(
                expert_id=expert_id,
                expert_type=expert_type,
                traits=(row or {}).get("traits") if isinstance((row or {}).get("traits"), dict) else None,
                assertiveness_fallback=float(
                    (row or {}).get("assertiveness")
                    if (row or {}).get("assertiveness") is not None
                    else 0.6
                ),
            )
            entries[expert_id] = RosterEntry(
                expert_id=expert_id,
                expert_type=expert_type,
                state=state,
                assertiveness=float((row or {}).get("assertiveness") if (row or {}).get("assertiveness") is not None else 0.6),
                traits=traits,
                personalities=_normalize_personalities(
                    expert_id=expert_id,
                    expert_type=expert_type,
                    base_traits=traits,
                    personalities=(row or {}).get("personalities") if isinstance((row or {}).get("personalities"), list) else None,
                ),
                offline_score=float((row or {}).get("offline_score") or 0.0),
                online_task_outcome=float((row or {}).get("online_task_outcome") or 0.0),
                online_sample_count=int((row or {}).get("online_sample_count") or 0),
                updated_at=str((row or {}).get("updated_at") or ""),
            )
        roster = cls(entries)
        roster._ensure_defaults(specialist_ids=specialist_ids)
        return roster

    def _ensure_defaults(self, *, specialist_ids: Iterable[str]) -> None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for expert_id in ("wide_compressed", "precise_short", "sliding_window"):
            if expert_id not in self.entries:
                traits = _default_traits_for(expert_id, "generalist_profile")
                self.entries[expert_id] = RosterEntry(
                    expert_id=expert_id,
                    expert_type="generalist_profile",
                    state="active",
                    assertiveness=traits["assertiveness"],
                    traits=traits,
                    personalities=_default_personalities_for(expert_id, "generalist_profile"),
                    updated_at=now,
                )
        for expert_id in sorted(set(str(x).strip() for x in specialist_ids if str(x).strip())):
            if not expert_id or expert_id == "general_fallback":
                continue
            if expert_id not in self.entries:
                traits = _default_traits_for(expert_id, "specialist_adapter")
                self.entries[expert_id] = RosterEntry(
                    expert_id=expert_id,
                    expert_type="specialist_adapter",
                    state="candidate",
                    assertiveness=traits["assertiveness"],
                    traits=traits,
                    personalities=_default_personalities_for(expert_id, "specialist_adapter"),
                    updated_at=now,
                )

    def assertiveness_map(self) -> dict[str, float]:
        return {
            expert_id: _normalize_traits(
                expert_id=expert_id,
                expert_type=entry.expert_type,
                traits=entry.traits,
                assertiveness_fallback=entry.assertiveness,
            )["assertiveness"]
            for expert_id, entry in self.entries.items()
        }

    def traits_map(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for expert_id, entry in self.entries.items():
            out[expert_id] = _normalize_traits(
                expert_id=expert_id,
                expert_type=entry.expert_type,
                traits=entry.traits,
                assertiveness_fallback=entry.assertiveness,
            )
        return out

    def personalities_map(self) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for expert_id, entry in self.entries.items():
            traits = _normalize_traits(
                expert_id=expert_id,
                expert_type=entry.expert_type,
                traits=entry.traits,
                assertiveness_fallback=entry.assertiveness,
            )
            out[expert_id] = _normalize_personalities(
                expert_id=expert_id,
                expert_type=entry.expert_type,
                base_traits=traits,
                personalities=entry.personalities,
            )
            out[expert_id] = [
                personality
                for personality in out[expert_id]
                if str(personality.get("state") or "") != "retired"
            ]
        return out

    def active_specialists(self) -> set[str]:
        out: set[str] = set()
        for entry in self.entries.values():
            if entry.expert_type != "specialist_adapter":
                continue
            if entry.state == "active":
                out.add(entry.expert_id)
        return out

    def apply_combined_gate(
        self,
        *,
        offline_scores: dict[str, float],
        online_scores: dict[str, float],
        online_counts: dict[str, int],
        personality_offline_scores: dict[str, dict[str, float]] | None = None,
        personality_offline_counts: dict[str, dict[str, int]] | None = None,
        personality_online_scores: dict[str, dict[str, float]] | None = None,
        personality_online_counts: dict[str, dict[str, int]] | None = None,
        min_offline_score: float,
        min_online_task_outcome: float,
        min_online_samples: int,
    ) -> dict[str, Any]:
        transitions: list[dict[str, Any]] = []
        personality_transitions: list[dict[str, Any]] = []
        personality_offline_scores = personality_offline_scores or {}
        personality_offline_counts = personality_offline_counts or {}
        personality_online_scores = personality_online_scores or {}
        personality_online_counts = personality_online_counts or {}
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for expert_id, entry in sorted(self.entries.items()):
            offline = float(offline_scores.get(expert_id, entry.offline_score))
            online = float(online_scores.get(expert_id, entry.online_task_outcome))
            count = int(online_counts.get(expert_id, entry.online_sample_count))
            offline_pass = offline >= min_offline_score
            online_seen = expert_id in online_scores
            online_pass = count >= min_online_samples and online >= min_online_task_outcome
            combined_pass = offline_pass and (online_pass if online_seen else True)
            previous = entry.state
            if entry.expert_type == "generalist_profile":
                # Keep generalists active unless they fail both gates hard.
                if not combined_pass and previous == "active":
                    entry.state = "probation"
                elif combined_pass:
                    entry.state = "active"
            else:
                if previous == "candidate":
                    entry.state = "active" if combined_pass else "candidate"
                elif previous == "active":
                    entry.state = "active" if combined_pass else "probation"
                elif previous == "probation":
                    entry.state = "active" if combined_pass else "demoted"
                else:
                    entry.state = "probation" if combined_pass else "demoted"

            entry.offline_score = offline
            entry.online_task_outcome = online
            entry.online_sample_count = count
            norm_traits = _normalize_traits(
                expert_id=expert_id,
                expert_type=entry.expert_type,
                traits=entry.traits,
                assertiveness_fallback=entry.assertiveness,
            )
            entry.assertiveness = norm_traits["assertiveness"]
            entry.traits = norm_traits
            entry.personalities = _normalize_personalities(
                expert_id=expert_id,
                expert_type=entry.expert_type,
                base_traits=norm_traits,
                personalities=entry.personalities,
            )
            updated_personalities: list[dict[str, Any]] = []
            for personality in entry.personalities:
                name = str(personality.get("name") or "").strip()
                previous_personality_state = str(
                    personality.get("state") or _personality_default_state(entry.expert_type)
                )
                p_offline = float(
                    (personality_offline_scores.get(expert_id) or {}).get(
                        name,
                        personality.get("offline_score", 0.0),
                    )
                    or 0.0
                )
                p_online = float(
                    (personality_online_scores.get(expert_id) or {}).get(
                        name,
                        personality.get("online_task_outcome", 0.0),
                    )
                    or 0.0
                )
                p_count = int(
                    (personality_online_counts.get(expert_id) or {}).get(
                        name,
                        personality.get("online_sample_count", 0),
                    )
                    or 0
                )
                p_offline_count = int(
                    (personality_offline_counts.get(expert_id) or {}).get(
                        name,
                        personality.get("offline_sample_count", 0),
                    )
                    or 0
                )
                p_offline_seen = name in (personality_offline_scores.get(expert_id) or {})
                p_online_seen = name in (personality_online_scores.get(expert_id) or {})
                if p_offline_seen or p_online_seen:
                    p_offline_pass = p_offline >= min_offline_score
                    p_online_pass = (
                        p_count >= min_online_samples
                        and p_online >= min_online_task_outcome
                    )
                    p_combined_pass = p_offline_pass and (p_online_pass if p_online_seen else True)
                    if previous_personality_state == "candidate":
                        next_personality_state = "active" if p_combined_pass else "candidate"
                    elif previous_personality_state == "active":
                        next_personality_state = "active" if p_combined_pass else "cooldown"
                    elif previous_personality_state == "cooldown":
                        next_personality_state = "active" if p_combined_pass else "retired"
                    else:
                        next_personality_state = "cooldown" if p_combined_pass else "retired"
                    if next_personality_state != previous_personality_state:
                        personality_transitions.append(
                            {
                                "expert_id": expert_id,
                                "personality": name,
                                "from": previous_personality_state,
                                "to": next_personality_state,
                                "offline_pass": p_offline_pass,
                                "online_pass": p_online_pass if p_online_seen else None,
                            }
                        )
                    personality["state"] = next_personality_state
                    personality["offline_score"] = p_offline
                    personality["offline_sample_count"] = p_offline_count
                    personality["online_task_outcome"] = p_online
                    personality["online_sample_count"] = p_count
                updated_personalities.append(personality)
            entry.personalities = updated_personalities
            entry.updated_at = now
            if previous != entry.state:
                transitions.append(
                    {
                        "expert_id": expert_id,
                        "from": previous,
                        "to": entry.state,
                        "offline_pass": offline_pass,
                        "online_pass": online_pass if online_seen else None,
                    }
                )
        return {
            "schema_version": "router_council_roster_transition_v1",
            "transitions": transitions,
            "personality_transitions": personality_transitions,
            "counts": {
                "active": sum(1 for e in self.entries.values() if e.state == "active"),
                "probation": sum(1 for e in self.entries.values() if e.state == "probation"),
                "candidate": sum(1 for e in self.entries.values() if e.state == "candidate"),
                "demoted": sum(1 for e in self.entries.values() if e.state == "demoted"),
            },
            "personality_counts": {
                state: sum(
                    1
                    for e in self.entries.values()
                    for p in e.personalities
                    if str(p.get("state") or "") == state
                )
                for state in sorted(VALID_PERSONALITY_STATES)
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "router_council_roster_v1",
            "entries": [row.to_dict() for _, row in sorted(self.entries.items())],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
