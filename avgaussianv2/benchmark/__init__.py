"""Strict benchmark protocol helpers."""

from avgaussianv2.benchmark.assets import (
    AssetAuditError,
    audit_audiogs_conversion,
    audit_ftgspp_flow_cache,
    audit_ftgspp_seed_record,
    audit_ftgspp_train_source,
    audit_ftgspp_upstream_config,
    audit_initialization_provenance,
    audit_protocol_config,
    prepare_fresh_ftgspp_namespaces,
    render_ftgspp_config,
)
from avgaussianv2.benchmark.output import (
    BenchmarkOutputError,
    BenchmarkOutputLock,
)
from avgaussianv2.benchmark.evaluation import (
    BenchmarkEvaluationError,
    BenchmarkEvaluationResult,
    BenchmarkEvaluator,
    BenchmarkPrediction,
    EvaluationIdentity,
    TrainingEvidence,
    load_evaluation,
    verify_evaluation,
)
from avgaussianv2.benchmark.report import (
    BenchmarkReportError,
    build_scene_report,
    build_suite_report,
    verify_scene_report,
    verify_suite_report,
)
from avgaussianv2.benchmark.runtime import (
    BenchmarkRuntime,
    build_production_runtime,
)
from avgaussianv2.benchmark.training import (
    BenchmarkCompatibility,
    BenchmarkConfig,
    BenchmarkMode,
    BenchmarkResumeError,
    FixedBudgetTrainer,
    build_worker_manifest,
    hash_shared_indices,
    make_shared_indices,
)

__all__ = [
    "AssetAuditError",
    "BenchmarkCompatibility",
    "BenchmarkConfig",
    "BenchmarkEvaluationError",
    "BenchmarkEvaluationResult",
    "BenchmarkEvaluator",
    "BenchmarkMode",
    "BenchmarkOutputError",
    "BenchmarkOutputLock",
    "BenchmarkPrediction",
    "BenchmarkReportError",
    "BenchmarkResumeError",
    "BenchmarkRuntime",
    "FixedBudgetTrainer",
    "EvaluationIdentity",
    "TrainingEvidence",
    "audit_audiogs_conversion",
    "audit_ftgspp_flow_cache",
    "audit_ftgspp_seed_record",
    "audit_ftgspp_train_source",
    "audit_ftgspp_upstream_config",
    "audit_initialization_provenance",
    "audit_protocol_config",
    "build_production_runtime",
    "build_scene_report",
    "build_suite_report",
    "build_worker_manifest",
    "hash_shared_indices",
    "make_shared_indices",
    "load_evaluation",
    "prepare_fresh_ftgspp_namespaces",
    "render_ftgspp_config",
    "verify_evaluation",
    "verify_scene_report",
    "verify_suite_report",
]
