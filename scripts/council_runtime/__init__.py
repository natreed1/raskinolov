"""Reusable council execution runtime.

UI, offline evaluation, and future season runners should call this package
instead of embedding their own round/participant/adjudication loops.
"""

from .executor import (
    CouncilGeneration,
    CouncilTurn,
    build_detailed_participant_prompt,
    build_eval_participant_prompt,
    run_council,
)
from .adapter_training import (
    ADAPTER_TRAIN_RESULT_SCHEMA,
    AdapterTrainResult,
    DryRunAdapterTrainingBackend,
    PPOAdapterTrainingBackend,
)
from .archetypes import (
    CouncilArchetypeEntry,
    CouncilArchetypeRegistry,
    TrainingLaneContract,
)
from .coevolution import (
    ArchetypePopulation,
    CoevolutionState,
    LoRAOrganism,
    PhaseResult,
    RewardMatrix,
    RewardPopulation,
    SeasonManifest,
    bootstrap_state,
    default_reward_population,
)
from .coevolution_runner import run_dry_season
from .planner_plan import (
    PLANNER_ROLLOUT_PLAN_SCHEMA,
    PlannerRolloutPlan,
    derive_planner_document,
    validate_planner_document,
)
from .fitness import (
    POPULATION_FITNESS_SCHEMA,
    FitnessAggregator,
    OrganismFitness,
    PopulationFitnessReport,
)
from .population_update import PopulationUpdater
from .pbt_selection import (
    PBTOrganismRank,
    PBTSelection,
    PBTSelectionConfig,
    select_all_populations_for_pbt,
    select_population_for_pbt,
)
from .reward_functions import (
    GRADED_ROLLOUT_SCHEMA,
    ROLLOUT_REWARD_SUMMARY_SCHEMA,
    PlannerRewardFunction,
    PlannerRewardResult,
    grade_trace_file,
    grade_rollouts,
)
from .rollouts import CouncilRollout, group_traces_into_rollouts
from .season_ops import (
    CouncilSeasonOperator,
    GradedRolloutBatch,
    JsonlTraceRolloutSource,
    PlannerRolloutGrader,
    RolloutBatch,
    SeasonContext,
    SeasonRunResult,
)
from .specialist_rewards import (
    SPECIALIST_REWARD_SCHEMA,
    SPECIALIST_REWARD_SUMMARY_SCHEMA,
    SpecialistRewardFunction,
    SpecialistRewardResult,
    SpecialistRewardSummary,
    grade_specialist_rollouts,
    read_specialist_reward_jsonl,
    summarize_specialist_rewards,
    write_specialist_rewards,
)
from .train_requests import (
    COUNCIL_LORA_TRAIN_REQUEST_SCHEMA,
    CouncilLoRATrainRequest,
    TrainRequestWriter,
    build_train_requests,
    specialist_rewards_to_ppo_scored_rows,
    write_ppo_scored_rows,
)
from .trace_scoring import (
    ScoredTrace,
    TraceScoreSummary,
    score_trace_file,
    score_trace_records,
)
from .traces import (
    CouncilTraceRecord,
    JsonlTraceWriter,
    TRACE_SCHEMAS,
)

__all__ = [
    "ADAPTER_TRAIN_RESULT_SCHEMA",
    "AdapterTrainResult",
    "ArchetypePopulation",
    "COUNCIL_LORA_TRAIN_REQUEST_SCHEMA",
    "CoevolutionState",
    "CouncilArchetypeEntry",
    "CouncilLoRATrainRequest",
    "CouncilRollout",
    "CouncilSeasonOperator",
    "CouncilTraceRecord",
    "CouncilArchetypeRegistry",
    "CouncilGeneration",
    "CouncilTurn",
    "DryRunAdapterTrainingBackend",
    "FitnessAggregator",
    "GRADED_ROLLOUT_SCHEMA",
    "GradedRolloutBatch",
    "JsonlTraceWriter",
    "JsonlTraceRolloutSource",
    "LoRAOrganism",
    "OrganismFitness",
    "PhaseResult",
    "PBTOrganismRank",
    "PBTSelection",
    "PBTSelectionConfig",
    "PlannerRewardFunction",
    "PlannerRewardResult",
    "PlannerRolloutPlan",
    "PLANNER_ROLLOUT_PLAN_SCHEMA",
    "POPULATION_FITNESS_SCHEMA",
    "PopulationFitnessReport",
    "PopulationUpdater",
    "PlannerRolloutGrader",
    "ROLLOUT_REWARD_SUMMARY_SCHEMA",
    "RewardMatrix",
    "RewardPopulation",
    "RolloutBatch",
    "ScoredTrace",
    "SeasonManifest",
    "SeasonContext",
    "SeasonRunResult",
    "SPECIALIST_REWARD_SCHEMA",
    "SPECIALIST_REWARD_SUMMARY_SCHEMA",
    "SpecialistRewardFunction",
    "SpecialistRewardResult",
    "SpecialistRewardSummary",
    "PPOAdapterTrainingBackend",
    "TRACE_SCHEMAS",
    "TraceScoreSummary",
    "TrainRequestWriter",
    "TrainingLaneContract",
    "bootstrap_state",
    "build_train_requests",
    "build_detailed_participant_prompt",
    "build_eval_participant_prompt",
    "default_reward_population",
    "derive_planner_document",
    "grade_specialist_rollouts",
    "grade_rollouts",
    "grade_trace_file",
    "group_traces_into_rollouts",
    "read_specialist_reward_jsonl",
    "run_council",
    "run_dry_season",
    "score_trace_file",
    "score_trace_records",
    "select_all_populations_for_pbt",
    "select_population_for_pbt",
    "specialist_rewards_to_ppo_scored_rows",
    "summarize_specialist_rewards",
    "validate_planner_document",
    "write_ppo_scored_rows",
    "write_specialist_rewards",
]
