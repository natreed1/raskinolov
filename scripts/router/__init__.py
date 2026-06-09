"""Router modules: classifier, council, and policy helpers."""

from .classifier import LinearRouterClassifier, Prediction
from .council import (
    CouncilAdjudication,
    CouncilPlan,
    CouncilParticipant,
    GENERALIST_PROFILES,
    adjudicate_council_outputs,
    build_council_plan,
    estimate_disagreement,
    shape_prompt_for_profile,
)
from .policy import derive_route_from_adapter, infer_coarse_adapter_candidates
from .roster import ExpertRoster, RosterEntry

__all__ = [
    "LinearRouterClassifier",
    "Prediction",
    "derive_route_from_adapter",
    "infer_coarse_adapter_candidates",
    "CouncilParticipant",
    "CouncilPlan",
    "CouncilAdjudication",
    "GENERALIST_PROFILES",
    "build_council_plan",
    "estimate_disagreement",
    "shape_prompt_for_profile",
    "ExpertRoster",
    "RosterEntry",
    "adjudicate_council_outputs",
]

