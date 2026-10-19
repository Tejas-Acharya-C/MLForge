"""MLForge Performance Benchmarking Runner.

Executes deterministic, synthetic tabular benchmark scenarios to measure
baseline timing and memory performance across dataset preparation, model evaluation,
diagnostics, and artifact serialization.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time
import tracemalloc
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path for direct CLI script execution
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
import scipy
from sklearn.datasets import make_classification
import sklearn

from mlforge.application.artifacts import serialize_run_artifacts
from mlforge.application.packaging import (
    build_audit_bundle,
    extract_audit_bundle,
    verify_audit_bundle,
)
from mlforge.application.persistence import (
    SQLiteRunRepository,
    execute_managed_evaluation,
)
from mlforge.application.release_certificates import (
    generate_release_keypair,
    issue_release_certificate,
    serialize_public_key,
    verify_release_certificate,
)
from mlforge.application.reporting import generate_html_report
from mlforge.core.contracts import (
    AcceptancePolicy,
    AcceptanceRule,
    CriterionSeverity,
    ReleasePolicy,
    RuleOperator,
)
from mlforge.engine.acceptance import evaluate_acceptance_policy
from mlforge.engine.evaluator import evaluate_classification
from mlforge.engine.pipeline import build_fold_pipeline
from mlforge.engine.robustness import (
    create_category_dropout_stress_profile,
    create_compound_stress_profile,
    create_missingness_stress_profile,
    create_noise_stress_profile,
    evaluate_model_robustness,
    evaluate_stress_profile,
    get_standard_stress_suite,
)


def generate_synthetic_churn_dataset(
    n_samples: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a realistic synthetic tabular churn dataset with mixed features and missing values."""
    rng = np.random.RandomState(random_state)
    X_num, y_raw = make_classification(
        n_samples=n_samples,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        weights=[0.70, 0.30],
        random_state=random_state,
    )
    df = pd.DataFrame(X_num, columns=[f"tenure_num_{i}" for i in range(4)])

    # Introduce controlled missing values (~5%) into numeric feature 1
    mask_num = rng.rand(n_samples) < 0.05
    df.loc[mask_num, "tenure_num_1"] = np.nan

    # Add categorical feature 1 (contract tier: 3 categories + missing values)
    contract_choices = ["month-to-month", "one-year", "two-year"]
    cats = rng.choice(contract_choices, size=n_samples, p=[0.55, 0.25, 0.20])
    df["contract_tier"] = cats
    mask_cat = rng.rand(n_samples) < 0.03
    df.loc[mask_cat, "contract_tier"] = None

    # Add categorical feature 2 (customer region: 4 segments)
    region_choices = ["North", "South", "East", "West"]
    df["region"] = rng.choice(region_choices, size=n_samples, p=[0.30, 0.30, 0.20, 0.20])

    df["churn"] = y_raw
    return df


def generate_breast_cancer_dataset(
    n_samples: int = 569,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic continuous-feature tabular dataset modeling breast cancer characteristics."""
    X_num, y_raw = make_classification(
        n_samples=n_samples,
        n_features=30,
        n_informative=15,
        n_redundant=10,
        n_classes=2,
        weights=[0.627, 0.373],  # Matching typical 357 / 212 distribution
        random_state=random_state,
    )
    feature_names = [f"mean_radius_{i}" for i in range(10)] + \
                    [f"se_radius_{i}" for i in range(10)] + \
                    [f"worst_radius_{i}" for i in range(10)]
    df = pd.DataFrame(X_num, columns=feature_names)
    df["diagnosis"] = y_raw
    return df


def generate_iris_multiclass_dataset(
    n_samples: int = 150,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic multiclass tabular dataset modeling 3-class iris characteristics."""
    X_num, y_raw = make_classification(
        n_samples=n_samples,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=3,
        weights=[0.333, 0.333, 0.334],
        random_state=random_state,
    )
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    df = pd.DataFrame(X_num, columns=feature_names)
    df["species"] = [f"class_{c}" for c in y_raw]
    return df


def _compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics (median, mean, min, max, std) for a list of measurements."""
    if not values:
        return {"median": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    return {
        "median": round(float(statistics.median(values)), 3),
        "mean": round(float(statistics.mean(values)), 3),
        "min": round(float(min(values)), 3),
        "max": round(float(max(values)), 3),
        "std": round(float(statistics.stdev(values)) if len(values) > 1 else 0.0, 3),
    }


def benchmark_scenario(
    scenario_id: str,
    name: str,
    dataset_generator,
    target_column: str,
    num_folds: int = 5,
    subgroup_columns: Optional[List[str]] = None,
    acceptance_policy: Optional[AcceptancePolicy] = None,
    repeats: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute a benchmark scenario across multiple deterministic repeats with isolated timings."""
    prep_times_ms: List[float] = []
    eval_times_ms: List[float] = []
    diag_times_ms: List[float] = []
    serial_times_ms: List[float] = []
    total_times_ms: List[float] = []
    peak_memory_kbs: List[float] = []

    dataset_meta_sample = None
    candidate_summary_sample = None
    status = "SUCCESS"
    error_message = None

    for i in range(repeats):
        tracemalloc.start()
        try:
            t_start = time.perf_counter()

            # 1. Dataset Preparation Timing
            t0 = time.perf_counter()
            df = dataset_generator(random_state=random_state)
            t_prep = (time.perf_counter() - t0) * 1000.0

            # 2. Core Evaluation Timing (Folds + Baseline vs Candidate)
            t0 = time.perf_counter()
            eval_res = evaluate_classification(
                df=df,
                target_column=target_column,
                num_folds=num_folds,
                random_state=random_state,
                subgroup_columns=subgroup_columns,
                acceptance_policy=acceptance_policy,
            )
            t_eval = (time.perf_counter() - t0) * 1000.0

            # 3. Dedicated Diagnostics Timing (Threshold + Calibration + Subgroups)
            # Evaluator runs diagnostics during fold execution; measure OOF calibration & threshold re-inspection
            t0 = time.perf_counter()
            _ = eval_res.candidate_result.calibration
            _ = eval_res.candidate_result.threshold_analysis
            _ = eval_res.candidate_result.error_diagnostics
            t_diag = (time.perf_counter() - t0) * 1000.0

            # 4. Artifact Serialization Timing (Atomic file writes & hashing)
            with tempfile.TemporaryDirectory() as tmp_dir:
                t0 = time.perf_counter()
                serialize_run_artifacts(
                    evaluation=eval_res,
                    acceptance_decision=eval_res.acceptance_decision,
                    artifacts_dir=Path(tmp_dir),
                    run_id=f"bench_{scenario_id}_rep_{i}",
                    policy=acceptance_policy,
                )
                t_serial = (time.perf_counter() - t0) * 1000.0

            t_total = (time.perf_counter() - t_start) * 1000.0
            _, peak_mem_bytes = tracemalloc.get_traced_memory()
            peak_mem_kb = peak_mem_bytes / 1024.0

            prep_times_ms.append(t_prep)
            eval_times_ms.append(t_eval)
            diag_times_ms.append(t_diag)
            serial_times_ms.append(t_serial)
            total_times_ms.append(t_total)
            peak_memory_kbs.append(peak_mem_kb)

            if dataset_meta_sample is None:
                dataset_meta_sample = {
                    "num_rows": eval_res.dataset_metadata.num_rows,
                    "num_features": eval_res.dataset_metadata.num_features,
                    "numeric_features": eval_res.dataset_metadata.numeric_feature_count,
                    "categorical_features": eval_res.dataset_metadata.categorical_feature_count,
                    "class_distribution": eval_res.dataset_metadata.class_distribution,
                    "checksum_sha256": eval_res.dataset_metadata.checksum_sha256,
                }
                cand_metrics = eval_res.candidate_result.metric_summaries
                candidate_summary_sample = {
                    "accuracy_mean": cand_metrics.get("accuracy").mean if "accuracy" in cand_metrics else None,
                    "macro_f1_mean": cand_metrics.get("macro_f1").mean if "macro_f1" in cand_metrics else None,
                    "macro_f1_ci95": cand_metrics.get("macro_f1").confidence_interval_95 if "macro_f1" in cand_metrics else None,
                }

        except Exception as exc:
            status = "FAILED"
            error_message = f"{type(exc).__name__}: {str(exc)}"
            break
        finally:
            tracemalloc.stop()

    return {
        "scenario_id": scenario_id,
        "name": name,
        "status": status,
        "error_message": error_message,
        "repeats": len(total_times_ms),
        "dataset_metadata": dataset_meta_sample,
        "candidate_summary": candidate_summary_sample,
        "timings_ms": {
            "dataset_preparation": _compute_stats(prep_times_ms),
            "model_evaluation": _compute_stats(eval_times_ms),
            "diagnostics_inspection": _compute_stats(diag_times_ms),
            "artifact_serialization": _compute_stats(serial_times_ms),
            "total_elapsed": _compute_stats(total_times_ms),
        },
        "peak_memory_kb": _compute_stats(peak_memory_kbs),
    }


def run_all_benchmarks(
    output_dir: Optional[Path] = None,
    repeats: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute all defined benchmark scenarios and serialize structured results to JSON."""
    timestamp = datetime.now(timezone.utc).isoformat()

    standard_policy = AcceptancePolicy(
        policy_id="benchmark_baseline_policy",
        version="1.0.0",
        name="Benchmark Policy",
        rules=[
            AcceptanceRule(
                rule_id="r_f1",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.60,
                severity=CriterionSeverity.BLOCKER,
            ),
        ],
    )

    scenarios = [
        # Scenario 1: Mixed tabular binary classification with missingness & subgroups
        benchmark_scenario(
            scenario_id="churn_binary_mixed",
            name="Synthetic Binary Churn (1000 rows, mixed features, missingness, 5-fold CV)",
            dataset_generator=generate_synthetic_churn_dataset,
            target_column="churn",
            num_folds=5,
            subgroup_columns=["region"],
            acceptance_policy=standard_policy,
            repeats=repeats,
            random_state=random_state,
        ),
        # Scenario 2: High-dimensional continuous binary classification
        benchmark_scenario(
            scenario_id="breast_cancer_continuous",
            name="Synthetic Breast Cancer (569 rows, 30 continuous features, 5-fold CV)",
            dataset_generator=generate_breast_cancer_dataset,
            target_column="diagnosis",
            num_folds=5,
            acceptance_policy=standard_policy,
            repeats=repeats,
            random_state=random_state,
        ),
        # Scenario 3: Multiclass classification
        benchmark_scenario(
            scenario_id="iris_multiclass",
            name="Synthetic Multiclass Iris (150 rows, 4 features, 3 balanced classes, 5-fold CV)",
            dataset_generator=generate_iris_multiclass_dataset,
            target_column="species",
            num_folds=5,
            repeats=repeats,
            random_state=random_state,
        ),
    ]

    report = {
        "benchmark_suite": "MLForge Classification Engine Performance Baseline",
        "milestone": "2.6",
        "timestamp": timestamp,
        "environment": {
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "scikit_learn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "pandas_version": pd.__version__,
        },
        "configuration": {
            "repeats": repeats,
            "random_state": random_state,
            "timing_method": "time.perf_counter() (monotonic, high-resolution)",
            "memory_profiler": "tracemalloc standard library peak traced memory",
        },
        "scenarios": scenarios,
        "limitations": [
            "Benchmarks reflect local single-process execution without hardware acceleration.",
            "Timings are environment-specific baseline measurements; no arbitrary performance gate is enforced.",
            "Regression benchmarking is omitted because the engine implements classification evaluation in Phase 2.",
        ],
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / "benchmark_results.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


def benchmark_robustness_scenario(
    scenario_id: str,
    name: str,
    dataset_generator,
    target_column: str,
    repeats: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute a stress testing benchmark scenario measuring timings and memory overhead across phases."""
    clean_eval_times_ms: List[float] = []
    noise_times_ms: List[float] = []
    missing_times_ms: List[float] = []
    dropout_times_ms: List[float] = []
    compound_times_ms: List[float] = []
    suite_times_ms: List[float] = []
    acceptance_times_ms: List[float] = []
    serial_times_ms: List[float] = []
    total_times_ms: List[float] = []
    peak_memory_kbs: List[float] = []

    status = "SUCCESS"
    error_message = None
    robustness_sample_summary = None

    stress_policy = AcceptancePolicy(
        policy_id="benchmark_robustness_policy",
        version="1.0.0",
        name="Benchmark Robustness Policy",
        rules=[
            AcceptanceRule(
                rule_id="rob_mean_aurdc",
                metric="mean_aurdc",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.65,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="robustness",
            ),
            AcceptanceRule(
                rule_id="rob_noise_max_drop",
                metric="gaussian_noise_max_drop_pct",
                operator=RuleOperator.LESS_THAN_OR_EQUAL,
                threshold=35.0,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="robustness",
            ),
        ],
    )

    for i in range(repeats):
        tracemalloc.start()
        try:
            t_start = time.perf_counter()

            # 1. Dataset Generation & Train/Validation Partitioning
            df = dataset_generator(random_state=random_state + i)
            y = df[target_column]
            X = df.drop(columns=[target_column])
            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = [c for c in X.columns if c not in numeric_features]

            n_val = max(1, int(len(df) * 0.20))
            X_train, X_val = X.iloc[:-n_val], X.iloc[-n_val:]
            y_train, y_val = y.iloc[:-n_val], y.iloc[-n_val:]

            pipeline = build_fold_pipeline(
                model_type="logistic_regression",
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                random_state=random_state,
            )
            pipeline.fit(X_train, y_train)

            # 2. Clean baseline evaluation timing
            t0 = time.perf_counter()
            clean_preds = pipeline.predict(X_val)
            _ = float(np.mean(clean_preds == y_val))
            t_clean = (time.perf_counter() - t0) * 1000.0

            # 3. Individual Stress Profiles
            # Noise
            t0 = time.perf_counter()
            p_noise = create_noise_stress_profile(target_columns=numeric_features, random_state=random_state)
            _ = evaluate_model_robustness(pipeline, X_val, y_val, [p_noise])
            t_noise = (time.perf_counter() - t0) * 1000.0

            # Missingness
            t0 = time.perf_counter()
            p_missing = create_missingness_stress_profile(random_state=random_state)
            _ = evaluate_model_robustness(pipeline, X_val, y_val, [p_missing])
            t_missing = (time.perf_counter() - t0) * 1000.0

            # Dropout
            t0 = time.perf_counter()
            if categorical_features:
                p_dropout = create_category_dropout_stress_profile(target_columns=categorical_features, random_state=random_state)
                _ = evaluate_model_robustness(pipeline, X_val, y_val, [p_dropout])
            t_dropout = (time.perf_counter() - t0) * 1000.0

            # Compound
            t0 = time.perf_counter()
            p_compound = create_compound_stress_profile(random_state=random_state)
            _ = evaluate_model_robustness(pipeline, X_val, y_val, [p_compound])
            t_compound = (time.perf_counter() - t0) * 1000.0

            # 4. Full Standard Stress Suite
            t0 = time.perf_counter()
            rob_result = evaluate_stress_profile(pipeline, X_val, y_val)
            t_suite = (time.perf_counter() - t0) * 1000.0

            # 5. Acceptance Policy Evaluation Timing
            eval_res = evaluate_classification(
                df=df,
                target_column=target_column,
                num_folds=3,
                random_state=random_state,
            )
            eval_res.candidate_result.robustness = rob_result

            t0 = time.perf_counter()
            decision = evaluate_acceptance_policy(eval_res, stress_policy)
            t_accept = (time.perf_counter() - t0) * 1000.0

            # 6. Artifact Serialization Timing (Atomic writes with full robustness curves)
            with tempfile.TemporaryDirectory() as tmp_dir:
                t0 = time.perf_counter()
                serialize_run_artifacts(
                    evaluation=eval_res,
                    acceptance_decision=decision,
                    artifacts_dir=Path(tmp_dir),
                    run_id=f"bench_rob_{scenario_id}_rep_{i}",
                    policy=stress_policy,
                )
                t_serial = (time.perf_counter() - t0) * 1000.0

            t_total = (time.perf_counter() - t_start) * 1000.0
            _, peak_mem_bytes = tracemalloc.get_traced_memory()
            peak_mem_kb = peak_mem_bytes / 1024.0

            clean_eval_times_ms.append(t_clean)
            noise_times_ms.append(t_noise)
            missing_times_ms.append(t_missing)
            dropout_times_ms.append(t_dropout)
            compound_times_ms.append(t_compound)
            suite_times_ms.append(t_suite)
            acceptance_times_ms.append(t_accept)
            serial_times_ms.append(t_serial)
            total_times_ms.append(t_total)
            peak_memory_kbs.append(peak_mem_kb)

            if robustness_sample_summary is None:
                robustness_sample_summary = {
                    "mean_aurdc": rob_result.summary_retention.get("mean_aurdc"),
                    "curves_evaluated": len(rob_result.curves),
                    "summary_retention_keys": sorted(list(rob_result.summary_retention.keys())),
                    "is_usable": rob_result.is_usable,
                }

        except Exception as exc:
            status = "FAILED"
            error_message = f"{type(exc).__name__}: {str(exc)}"
            break
        finally:
            tracemalloc.stop()

    return {
        "scenario_id": scenario_id,
        "name": name,
        "status": status,
        "error_message": error_message,
        "repeats": len(total_times_ms),
        "robustness_summary": robustness_sample_summary,
        "timings_ms": {
            "clean_baseline_evaluation": _compute_stats(clean_eval_times_ms),
            "noise_profile": _compute_stats(noise_times_ms),
            "missingness_profile": _compute_stats(missing_times_ms),
            "dropout_profile": _compute_stats(dropout_times_ms),
            "compound_profile": _compute_stats(compound_times_ms),
            "full_standard_suite": _compute_stats(suite_times_ms),
            "acceptance_policy_evaluation": _compute_stats(acceptance_times_ms),
            "artifact_serialization": _compute_stats(serial_times_ms),
            "total_elapsed": _compute_stats(total_times_ms),
        },
        "peak_traced_memory_kb": _compute_stats(peak_memory_kbs),
    }


def run_robustness_benchmarks(
    output_dir: Optional[Path] = None,
    repeats: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute robustness and stress testing performance benchmarks across scenarios."""
    timestamp = datetime.now(timezone.utc).isoformat()
    scenarios = [
        benchmark_robustness_scenario(
            scenario_id="churn_stress_testing",
            name="Synthetic Binary Churn Stress Testing (1000 rows, mixed features, full standard suite)",
            dataset_generator=generate_synthetic_churn_dataset,
            target_column="churn",
            repeats=repeats,
            random_state=random_state,
        ),
        benchmark_robustness_scenario(
            scenario_id="breast_cancer_stress_testing",
            name="Synthetic Breast Cancer Stress Testing (569 rows, 30 continuous features, full standard suite)",
            dataset_generator=generate_breast_cancer_dataset,
            target_column="diagnosis",
            repeats=repeats,
            random_state=random_state,
        ),
    ]

    report = {
        "benchmark_suite": "MLForge Robustness & Stress Testing Engine Performance Baseline",
        "milestone": "3.4",
        "timestamp": timestamp,
        "environment": {
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "scikit_learn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "pandas_version": pd.__version__,
        },
        "configuration": {
            "repeats": repeats,
            "random_state": random_state,
            "timing_method": "time.perf_counter() (monotonic, high-resolution)",
            "memory_profiler": "tracemalloc standard library peak traced memory",
        },
        "scenarios": scenarios,
        "limitations": [
            "Benchmarks reflect local single-process execution without GPU/hardware acceleration.",
            "Timings are environment-specific baseline measurements; no arbitrary performance gate is enforced.",
            "Memory measurements reflect Python heap allocations traced by tracemalloc, not total OS process RSS.",
            "Variability across runs is reported via standard deviation and min/max ranges.",
        ],
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / "robustness_benchmark_results.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


def benchmark_phase4_scenario(
    scenario_id: str,
    name: str,
    dataset_generator,
    target_column: str,
    num_folds: int = 3,
    subgroup_columns: Optional[List[str]] = None,
    repeats: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute a Phase 4 full lifecycle benchmark scenario measuring timings and memory overhead."""
    eval_persist_times_ms: List[float] = []
    report_times_ms: List[float] = []
    gate_times_ms: List[float] = []
    cert_issue_times_ms: List[float] = []
    bundle_build_times_ms: List[float] = []
    bundle_verify_times_ms: List[float] = []
    cert_verify_dir_times_ms: List[float] = []
    cert_verify_bundle_times_ms: List[float] = []
    extract_times_ms: List[float] = []
    total_times_ms: List[float] = []
    peak_memory_kbs: List[float] = []

    status = "SUCCESS"
    error_message = None
    sample_summary = None

    release_policy = ReleasePolicy(
        policy_id="bench_phase4_policy",
        version="1.0.0",
        target_environment="production",
        require_acceptance_pass=False,
        require_model_artifact=True,
        require_manifest_integrity=True,
        min_evaluated_folds=2,
    )

    for i in range(repeats):
        tracemalloc.start()
        try:
            t_start = time.perf_counter()

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                db_path = tmp_path / f"bench_{scenario_id}_{i}.db"
                repo = SQLiteRunRepository(db_path=db_path)
                repo.initialize_schema()

                df = dataset_generator(random_state=random_state + i)
                run_id = f"bench_{scenario_id}_rep_{i}"
                artifacts_dir = tmp_path / "artifacts"

                # 1. Managed Evaluation & SQLite Persistence
                t0 = time.perf_counter()
                record, eval_res, run_dir = execute_managed_evaluation(
                    df=df,
                    target_column=target_column,
                    num_folds=num_folds,
                    random_state=random_state,
                    subgroup_columns=subgroup_columns,
                    artifacts_dir=artifacts_dir,
                    repository=repo,
                    run_id=run_id,
                )
                t_eval_persist = (time.perf_counter() - t0) * 1000.0

                # 2. Offline HTML Report Generation
                t0 = time.perf_counter()
                report_file = generate_html_report(run_dir)
                t_report = (time.perf_counter() - t0) * 1000.0

                # 3. Release Gate Evaluation
                from mlforge.engine.release_gate import evaluate_release_gate
                t0 = time.perf_counter()
                gate_res = evaluate_release_gate(run_dir, policy=release_policy)
                t_gate = (time.perf_counter() - t0) * 1000.0

                # 4. Ed25519 Certificate Issuance
                priv_key, pub_key, key_id = generate_release_keypair()
                t0 = time.perf_counter()
                cert = issue_release_certificate(
                    source=run_dir,
                    private_key=priv_key,
                    key_id=key_id,
                    release_policy=release_policy,
                )
                t_cert_issue = (time.perf_counter() - t0) * 1000.0

                # 5. Audit Bundle Creation (Index certificate into manifest first)
                cert_file = run_dir / "release" / "release_certificate.json"
                manifest_file = run_dir / "manifest.json"
                mdata = json.loads(manifest_file.read_text(encoding="utf-8"))
                mdata["file_checksums"]["release/release_certificate.json"] = hashlib.sha256(cert_file.read_bytes()).hexdigest()
                manifest_file.write_text(json.dumps(mdata, indent=2), encoding="utf-8")

                bundle_zip = tmp_path / f"{run_id}_audit_bundle.zip"
                t0 = time.perf_counter()
                build_audit_bundle(run_dir=run_dir, output_bundle_path=bundle_zip)
                t_bundle_build = (time.perf_counter() - t0) * 1000.0

                # 6. Audit Bundle Verification (archive integrity & limits)
                t0 = time.perf_counter()
                bundle_verif = verify_audit_bundle(bundle_zip)
                t_bundle_verify = (time.perf_counter() - t0) * 1000.0

                # 7. Certificate Verification (directory vs bundle)
                trusted_registry = {key_id: pub_key}
                t0 = time.perf_counter()
                res_dir = verify_release_certificate(cert, trusted_keys=trusted_registry, expected_source=run_dir)
                t_cert_dir = (time.perf_counter() - t0) * 1000.0

                t0 = time.perf_counter()
                res_bundle = verify_release_certificate(cert, trusted_keys=trusted_registry, expected_source=bundle_zip)
                t_cert_bundle = (time.perf_counter() - t0) * 1000.0

                # 8. Safe Bundle Extraction
                extract_dest = tmp_path / "extracted_bundle"
                t0 = time.perf_counter()
                extract_audit_bundle(bundle_zip, destination_dir=extract_dest, verify_first=True)
                t_extract = (time.perf_counter() - t0) * 1000.0

            t_total = (time.perf_counter() - t_start) * 1000.0
            _, peak_mem_bytes = tracemalloc.get_traced_memory()
            peak_mem_kb = peak_mem_bytes / 1024.0

            eval_persist_times_ms.append(t_eval_persist)
            report_times_ms.append(t_report)
            gate_times_ms.append(t_gate)
            cert_issue_times_ms.append(t_cert_issue)
            bundle_build_times_ms.append(t_bundle_build)
            bundle_verify_times_ms.append(t_bundle_verify)
            cert_verify_dir_times_ms.append(t_cert_dir)
            cert_verify_bundle_times_ms.append(t_cert_bundle)
            extract_times_ms.append(t_extract)
            total_times_ms.append(t_total)
            peak_memory_kbs.append(peak_mem_kb)

            if sample_summary is None:
                sample_summary = {
                    "run_id": run_id,
                    "dataset_samples": len(df),
                    "num_folds": num_folds,
                    "bundle_verified": bundle_verif.is_valid,
                    "release_authorized": res_bundle.release_authorized,
                }

        except Exception as exc:
            status = "FAILED"
            error_message = f"{type(exc).__name__}: {str(exc)}"
            break
        finally:
            tracemalloc.stop()

    return {
        "scenario_id": scenario_id,
        "name": name,
        "status": status,
        "error_message": error_message,
        "repeats": len(total_times_ms),
        "workload_summary": sample_summary,
        "timings_ms": {
            "managed_evaluation_and_persistence": _compute_stats(eval_persist_times_ms),
            "offline_html_report_generation": _compute_stats(report_times_ms),
            "release_gate_evaluation": _compute_stats(gate_times_ms),
            "certificate_issuance": _compute_stats(cert_issue_times_ms),
            "audit_bundle_packaging": _compute_stats(bundle_build_times_ms),
            "audit_bundle_verification": _compute_stats(bundle_verify_times_ms),
            "certificate_verification_directory": _compute_stats(cert_verify_dir_times_ms),
            "certificate_verification_bundle": _compute_stats(cert_verify_bundle_times_ms),
            "audit_bundle_safe_extraction": _compute_stats(extract_times_ms),
            "total_elapsed": _compute_stats(total_times_ms),
        },
        "peak_traced_memory_kb": _compute_stats(peak_memory_kbs),
    }


def run_phase4_benchmarks(
    output_dir: Optional[Path] = None,
    repeats: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute Phase 4 lifecycle, persistence, security & packaging performance benchmarks."""
    timestamp = datetime.now(timezone.utc).isoformat()
    scenarios = [
        benchmark_phase4_scenario(
            scenario_id="churn_phase4_lifecycle",
            name="Synthetic Binary Churn Phase 4 Pipeline (500 rows, mixed features, 3-fold CV)",
            dataset_generator=lambda random_state=42: generate_synthetic_churn_dataset(n_samples=500, random_state=random_state),
            target_column="churn",
            subgroup_columns=["region"],
            num_folds=3,
            repeats=repeats,
            random_state=random_state,
        ),
        benchmark_phase4_scenario(
            scenario_id="breast_cancer_phase4_lifecycle",
            name="Synthetic Breast Cancer Phase 4 Pipeline (300 rows, 30 continuous features, 3-fold CV)",
            dataset_generator=lambda random_state=42: generate_breast_cancer_dataset(n_samples=300, random_state=random_state),
            target_column="diagnosis",
            num_folds=3,
            repeats=repeats,
            random_state=random_state,
        ),
    ]

    report = {
        "benchmark_suite": "MLForge Phase 4 Lifecycle, Persistence, Security & Packaging Baseline",
        "milestone": "4.5",
        "timestamp": timestamp,
        "environment": {
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "scikit_learn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "pandas_version": pd.__version__,
        },
        "configuration": {
            "repeats": repeats,
            "random_state": random_state,
            "timing_method": "time.perf_counter() (monotonic, high-resolution)",
            "memory_measurement_method": "tracemalloc peak Python heap allocation (excludes OS process RSS, native C/C++ buffers, and memory-mapped files)",
        },
        "scenarios": scenarios,
        "limitations": [
            "Benchmarks reflect local single-process execution without hardware acceleration.",
            "Timings are environment-specific baseline measurements; no arbitrary performance gate is enforced.",
            "Memory measurements reflect Python heap allocations traced by tracemalloc, not total OS process RSS.",
            "Variability across runs is reported via standard deviation and min/max ranges.",
        ],
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / "phase4_benchmark_results.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    out_dir = Path("artifacts/benchmarks")
    print(f"Running MLForge Milestone 2.6 Classification Benchmarks (5 repeats)...")
    res_26 = run_all_benchmarks(output_dir=out_dir, repeats=5)
    print(f"Classification benchmark results written to {out_dir / 'benchmark_results.json'}")
    for sc in res_26["scenarios"]:
        print(f"Scenario: {sc['scenario_id']} - {sc['name']}")
        print(f"  Status: {sc['status']}")
        t = sc["timings_ms"]
        print(f"  Total Elapsed (median): {t['total_elapsed']['median']} ms (std: {t['total_elapsed']['std']} ms)")
        print(f"  Prep: {t['dataset_preparation']['median']} ms | Eval: {t['model_evaluation']['median']} ms | Serial: {t['artifact_serialization']['median']} ms")
        print(f"  Peak Memory: {sc['peak_memory_kb']['median']} KB")

    print(f"\nRunning MLForge Milestone 3.4 Robustness & Stress Testing Benchmarks (5 repeats)...")
    res_34 = run_robustness_benchmarks(output_dir=out_dir, repeats=5)
    print(f"Robustness benchmark results written to {out_dir / 'robustness_benchmark_results.json'}")
    for sc in res_34["scenarios"]:
        print(f"Scenario: {sc['scenario_id']} - {sc['name']}")
        print(f"  Status: {sc['status']}")
        t = sc["timings_ms"]
        print(f"  Total Elapsed (median): {t['total_elapsed']['median']} ms (std: {t['total_elapsed']['std']} ms)")
        print(f"  Stress Suite: {t['full_standard_suite']['median']} ms | Acceptance: {t['acceptance_policy_evaluation']['median']} ms | Serial: {t['artifact_serialization']['median']} ms")
        print(f"  Peak Traced Memory: {sc['peak_traced_memory_kb']['median']} KB")

    print(f"\nRunning MLForge Milestone 4.5 Phase 4 Unified Benchmarks (3 repeats)...")
    res_45 = run_phase4_benchmarks(output_dir=out_dir, repeats=3)
    print(f"Phase 4 benchmark results written to {out_dir / 'phase4_benchmark_results.json'}")
    for sc in res_45["scenarios"]:
        print(f"Scenario: {sc['scenario_id']} - {sc['name']}")
        print(f"  Status: {sc['status']}")
        t = sc["timings_ms"]
        print(f"  Total Elapsed (median): {t['total_elapsed']['median']} ms (std: {t['total_elapsed']['std']} ms)")
        print(f"  Eval+Persist: {t['managed_evaluation_and_persistence']['median']} ms | Report: {t['offline_html_report_generation']['median']} ms | Gate: {t['release_gate_evaluation']['median']} ms")
        print(f"  Cert Issue: {t['certificate_issuance']['median']} ms | Bundle Build: {t['audit_bundle_packaging']['median']} ms | Bundle Verify: {t['audit_bundle_verification']['median']} ms")
        print(f"  Peak Traced Memory: {sc['peak_traced_memory_kb']['median']} KB")
