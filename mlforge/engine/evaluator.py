"""MLForge Classification Evaluation Orchestrator.

Orchestrates deterministic stratified cross-validation, leakage-safe pipeline fitting,
and baseline versus candidate model evaluation.
"""

import platform
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import scipy
import sklearn

from mlforge.core.contracts import (
    EvaluationResult,
    FoldResult,
    MetricSummary,
    ModelEvaluationResult,
    ReproducibilityMetadata,
)
from mlforge.core.dataset import validate_and_extract_dataset
from mlforge.engine.diagnostics import (
    aggregate_error_diagnostics,
    analyze_classification_thresholds,
    compute_fold_error_diagnostic,
    compute_subgroup_diagnostics,
)
from mlforge.engine.metrics import (
    aggregate_metric_folds,
    calculate_fold_classification_metrics,
)
from mlforge.engine.pipeline import build_fold_pipeline
from mlforge.engine.protocol import generate_stratified_folds
from mlforge.engine.reliability import compute_calibration_curve


def _evaluate_single_model(
    model_name: str,
    model_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list,
    numeric_features: List[str],
    categorical_features: List[str],
    class_labels: List[str],
    hyperparameters: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
    subgroup_columns: Optional[List[str]] = None,
    metadata_df: Optional[pd.DataFrame] = None,
) -> ModelEvaluationResult:
    """Evaluate a single model across generated folds with isolated fold pipelines."""
    fold_results: List[FoldResult] = []
    n_classes = len(class_labels)
    aggregate_cm = np.zeros((n_classes, n_classes), dtype=int)

    # Containers for out-of-fold pooled predictions
    all_oof_indices: List[int] = []
    all_oof_y_true: List[Any] = []
    all_oof_y_pred: List[Any] = []
    all_oof_y_proba: List[float] = []
    has_proba = True
    pos_label = class_labels[1] if n_classes == 2 else None
    last_fitted_pipeline = None

    for fold in folds:
        # 1. Slice train and validation sets strictly by fold indices
        X_train = X.iloc[fold.train_indices]
        y_train = y.iloc[fold.train_indices]
        X_val = X.iloc[fold.val_indices]
        y_val = y.iloc[fold.val_indices]

        # 2. Build fresh, unfitted pipeline instance
        pipeline = build_fold_pipeline(
            model_type=model_type,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            hyperparameters=hyperparameters,
            random_state=random_state,
        )

        # 3. Fit strictly on training partition
        t0_fit = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_duration = time.perf_counter() - t0_fit

        # 4. Predict on validation partition
        t0_pred = time.perf_counter()
        y_pred = pipeline.predict(X_val)
        y_proba = None
        if hasattr(pipeline, "predict_proba"):
            try:
                y_proba = pipeline.predict_proba(X_val)
            except Exception:
                y_proba = None
        predict_duration = time.perf_counter() - t0_pred

        # 5. Compute fold metrics
        metrics, cm = calculate_fold_classification_metrics(
            y_true=y_val.to_numpy(),
            y_pred=y_pred,
            y_proba=y_proba,
            class_labels=class_labels,
        )

        aggregate_cm += np.array(cm, dtype=int)

        # 6. Fold-level Error Diagnostic
        fold_err_diag = compute_fold_error_diagnostic(
            y_true=y_val.to_numpy(),
            y_pred=y_pred,
            fold_index=fold.fold_index,
            positive_label=pos_label,
        )

        # 7. Fold-level Calibration & Threshold Analysis (if binary and probabilities available)
        fold_calib = None
        fold_threshold = None
        if y_proba is not None and n_classes == 2 and pos_label is not None:
            try:
                fold_calib = compute_calibration_curve(
                    y_true=y_val.to_numpy(),
                    y_prob=y_proba,
                    positive_label=pos_label,
                )
                fold_threshold = analyze_classification_thresholds(
                    y_true=y_val.to_numpy(),
                    y_prob=y_proba,
                    positive_label=pos_label,
                )
            except Exception:
                fold_calib = None
                fold_threshold = None

        val_indices_list = fold.val_indices.tolist() if hasattr(fold.val_indices, "tolist") else list(fold.val_indices)

        fold_results.append(
            FoldResult(
                fold_index=fold.fold_index,
                train_indices_count=fold.train_count,
                val_indices_count=fold.val_count,
                metrics=metrics,
                confusion_matrix=cm,
                class_labels=class_labels,
                fit_duration_seconds=round(fit_duration, 5),
                predict_duration_seconds=round(predict_duration, 5),
                partition_fingerprint=fold.partition_fingerprint,
                calibration=fold_calib,
                threshold_analysis=fold_threshold,
                error_diagnostic=fold_err_diag,
                val_indices=val_indices_list,
                fitted_pipeline=pipeline,
            )
        )

        last_fitted_pipeline = pipeline

        # Collect OOF predictions
        all_oof_indices.extend(val_indices_list)
        all_oof_y_true.extend(y_val.to_list() if hasattr(y_val, "to_list") else list(y_val))
        all_oof_y_pred.extend(y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred))
        if y_proba is not None and n_classes == 2:
            # Positive class probability at column 1
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                all_oof_y_proba.extend(y_proba[:, 1].tolist())
            else:
                all_oof_y_proba.extend(y_proba.ravel().tolist())
        else:
            has_proba = False

    # 8. Aggregate metric summaries across folds
    metric_keys = list(fold_results[0].metrics.keys())
    metric_summaries: Dict[str, MetricSummary] = {}

    for k in metric_keys:
        fold_values = [fr.metrics[k] for fr in fold_results]
        metric_summaries[k] = aggregate_metric_folds(metric_name=k, fold_metric_values=fold_values)

    # 9. Out-of-fold Calibration, Threshold, and Subgroup Error Diagnostics
    oof_calibration = None
    oof_thresholds = None
    if has_proba and len(all_oof_y_proba) == len(all_oof_y_true) and n_classes == 2 and pos_label is not None:
        try:
            oof_calibration = compute_calibration_curve(
                y_true=all_oof_y_true,
                y_prob=all_oof_y_proba,
                positive_label=pos_label,
            )
            oof_thresholds = analyze_classification_thresholds(
                y_true=all_oof_y_true,
                y_prob=all_oof_y_proba,
                positive_label=pos_label,
            )
        except Exception:
            oof_calibration = None
            oof_thresholds = None

    # Subgroup breakdown if explicitly requested
    subgroup_diagnostics = None
    if subgroup_columns:
        source_df = metadata_df if metadata_df is not None else X
        # Re-order metadata rows to match OOF collection order
        oof_meta_df = source_df.iloc[all_oof_indices].reset_index(drop=True)
        try:
            subgroup_diagnostics = compute_subgroup_diagnostics(
                y_true=all_oof_y_true,
                y_pred=all_oof_y_pred,
                metadata_df=oof_meta_df,
                subgroup_columns=subgroup_columns,
                positive_label=pos_label,
            )
        except Exception:
            subgroup_diagnostics = None

    fold_diags = [fr.error_diagnostic for fr in fold_results if fr.error_diagnostic is not None]
    oof_error_diagnostics = aggregate_error_diagnostics(
        fold_diagnostics=fold_diags,
        total_samples=len(all_oof_y_true),
        subgroup_diagnostics=subgroup_diagnostics,
    )

    return ModelEvaluationResult(
        model_name=model_name,
        model_type=model_type,
        hyperparameters=hyperparameters or {},
        fold_results=fold_results,
        metric_summaries=metric_summaries,
        aggregate_confusion_matrix=aggregate_cm.tolist(),
        calibration=oof_calibration,
        threshold_analysis=oof_thresholds,
        error_diagnostics=oof_error_diagnostics,
        fitted_pipeline=last_fitted_pipeline,
        model_provenance="final_cv_fold" if last_fitted_pipeline is not None else None,
        fitted_fold_index=folds[-1].fold_index if (last_fitted_pipeline is not None and folds) else None,
    )


def evaluate_classification(
    df: pd.DataFrame,
    target_column: str,
    num_folds: int = 5,
    candidate_hyperparameters: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
    subgroup_columns: Optional[List[str]] = None,
    metadata_df: Optional[pd.DataFrame] = None,
    acceptance_policy: Optional[Any] = None,
) -> EvaluationResult:
    """Run an end-to-end, leakage-safe classification evaluation comparing candidate against baseline.

    Parameters
    ----------
    df : pd.DataFrame
        Input tabular dataset.
    target_column : str
        Target column name.
    num_folds : int, default=5
        Number of cross-validation folds (>= 2).
    candidate_hyperparameters : Optional[Dict[str, Any]]
        Optional hyperparameters for candidate LogisticRegression model.
    random_state : int, default=42
        Deterministic random seed.

    Returns
    -------
    EvaluationResult
        Complete evaluation contract with metadata, reproducibility info, and both model results.
    """
    # 1. Dataset validation and metadata extraction
    X, y, metadata = validate_and_extract_dataset(
        df=df,
        target_column=target_column,
        k_folds=num_folds,
    )

    # 2. Partition feature columns into numeric and categorical
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    # 3. Generate deterministic cross-validation folds once
    folds = generate_stratified_folds(
        X=X,
        y=y,
        num_folds=num_folds,
        shuffle=True,
        random_state=random_state,
    )

    # 4. Evaluate Baseline Model (DummyClassifier)
    baseline_result = _evaluate_single_model(
        model_name="Baseline: DummyClassifier",
        model_type="dummy_classifier",
        X=X,
        y=y,
        folds=folds,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        class_labels=metadata.class_labels,
        hyperparameters={"strategy": "prior"},
        random_state=random_state,
        subgroup_columns=subgroup_columns,
        metadata_df=metadata_df,
    )

    # 5. Evaluate Candidate Model (LogisticRegression)
    candidate_result = _evaluate_single_model(
        model_name="Candidate: LogisticRegression",
        model_type="logistic_regression",
        X=X,
        y=y,
        folds=folds,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        class_labels=metadata.class_labels,
        hyperparameters=candidate_hyperparameters,
        random_state=random_state,
        subgroup_columns=subgroup_columns,
        metadata_df=metadata_df,
    )

    # 6. Capture environment and reproducibility metadata
    reproducibility = ReproducibilityMetadata(
        random_seed=random_state,
        split_strategy="stratified_k_fold",
        num_folds=num_folds,
        package_versions={
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
        },
        platform_system=platform.platform(),
        python_version=sys.version.split()[0],
    )

    # 7. Perform paired statistical comparison between candidate and baseline
    from mlforge.core.exceptions import IncompatibleEvaluationPairError
    from mlforge.engine.statistics import compare_model_evaluations

    comparison_report: Optional[Any] = None
    comparison_warnings: List[str] = []

    try:
        comparison_report = compare_model_evaluations(
            candidate=candidate_result,
            baseline=baseline_result,
            dataset_metadata=metadata,
            reproducibility=reproducibility,
        )
        comparison_warnings = comparison_report.warnings
    except IncompatibleEvaluationPairError as exc:
        comparison_report = None
        comparison_warnings = [f"Statistical comparison could not be performed: {exc}"]

    combined_warnings = list(set(metadata.warnings + comparison_warnings))

    eval_result_initial = EvaluationResult(
        dataset_metadata=metadata,
        reproducibility=reproducibility,
        candidate_result=candidate_result,
        baseline_result=baseline_result,
        comparison_report=comparison_report,
        warnings=combined_warnings,
    )

    if acceptance_policy is not None:
        from mlforge.engine.acceptance import evaluate_acceptance_policy
        decision = evaluate_acceptance_policy(
            evaluation=eval_result_initial,
            policy=acceptance_policy,
        )
        return EvaluationResult(
            dataset_metadata=metadata,
            reproducibility=reproducibility,
            candidate_result=candidate_result,
            baseline_result=baseline_result,
            comparison_report=comparison_report,
            acceptance_decision=decision,
            warnings=combined_warnings,
        )

    return eval_result_initial
