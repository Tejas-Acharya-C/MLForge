"""MLForge Robustness & Stress Testing Engine (Milestone 3.1 & 3.2).

Provides mathematically rigorous, reproducible, non-mutating perturbation transformers,
multi-feature compound perturbations, pre-packaged stress profiles, AURDC calculation,
and robustness evaluation procedures for tabular models under controlled corruptions.
"""

import hashlib
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from mlforge.core.contracts import (
    MetricDirection,
    PerturbationConfig,
    PerturbationType,
    RobustnessCurve,
    RobustnessEvaluationResult,
    RobustnessPerturbationResult,
)
from mlforge.engine.metrics import calculate_fold_classification_metrics
from mlforge.engine.statistics import METRIC_DIRECTIONS

STANDARD_SEVERITY_GRID: List[float] = [0.05, 0.10, 0.25, 0.50]


def _derive_column_seed(random_state: int, idx: int, column_name: Any) -> int:
    """Derive a deterministic 31-bit random seed for a column independent of Python hash randomization."""
    col_str = str(column_name).encode("utf-8")
    digest = hashlib.sha256(col_str).hexdigest()
    col_hash = int(digest[:8], 16)
    return (int(random_state) + idx * 10007 + col_hash) % (2**31 - 1)


def apply_gaussian_noise(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    severity: float = 0.10,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, List[str]]:
    """Inject additive zero-mean Gaussian noise scaled to feature standard deviation.

    Noise model: x_ij_perturbed = x_ij + delta, delta ~ N(0, (severity * sigma_j)^2)
    where sigma_j is the sample standard deviation of feature j.

    Parameters
    ----------
    df : pd.DataFrame
        Input tabular feature DataFrame. Unmutated defensive copy is returned.
    columns : sequence of str, optional
        Target numeric columns to perturb. If None, all numeric columns with sigma > 0.
    severity : float, default=0.10
        Noise multiplier (severity >= 0.0). severity=0.10 represents 10% standard deviation.
    random_state : int, default=42
        Random seed for deterministic corruption.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        - Perturbed feature DataFrame (defensive copy, original unmutated).
        - Advisory warnings (e.g. constant columns skipped).

    Raises
    ------
    ValueError
        If df is empty, severity < 0 or non-finite, or columns contain non-numeric features.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected df to be a pandas DataFrame, got {type(df).__name__}.")
    if df.empty or len(df) == 0:
        raise ValueError("Cannot perturb an empty DataFrame.")
    if severity is None or np.isnan(severity) or np.isinf(severity) or severity < 0.0:
        raise ValueError(f"Severity must be a finite non-negative float (>= 0.0); received {severity}.")

    df_pert = df.copy(deep=True)
    warnings: List[str] = []

    if severity == 0.0:
        return df_pert, warnings

    all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if columns is not None:
        missing_cols = [c for c in columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Specified column(s) not found in DataFrame: {missing_cols}")
        non_num = [c for c in columns if c not in all_numeric_cols]
        if non_num:
            raise ValueError(
                f"Gaussian noise can only be applied to numeric columns. Non-numeric columns: {non_num}"
            )
        target_cols = list(columns)
    else:
        target_cols = all_numeric_cols

    if not target_cols:
        warnings.append("No numeric feature columns available for Gaussian noise injection.")
        return df_pert, warnings

    for idx, col in enumerate(target_cols):
        col_series = df[col]
        std_val = float(col_series.std(skipna=True))

        if np.isnan(std_val) or std_val == 0.0:
            warnings.append(f"Column '{col}' exhibits zero variance or all NaN; skipping Gaussian noise.")
            continue

        scale = severity * std_val
        col_seed = _derive_column_seed(random_state, idx, col)
        rng = np.random.RandomState(col_seed)

        # Generate noise matching column length
        noise = rng.normal(loc=0.0, scale=scale, size=len(df))

        # Preserve NaN locations
        valid_mask = ~col_series.isna()
        perturbed_values = col_series.to_numpy(dtype=float, copy=True)
        perturbed_values[valid_mask] += noise[valid_mask]
        df_pert[col] = perturbed_values

    return df_pert, warnings


def apply_missingness_injection(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    severity: float = 0.10,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, List[str]]:
    """Inject Missing Completely At Random (MCAR) feature dropouts.

    Parameters
    ----------
    df : pd.DataFrame
        Input tabular feature DataFrame. Unmutated defensive copy is returned.
    columns : sequence of str, optional
        Target columns to mask. If None, all feature columns are eligible.
    severity : float, default=0.10
        Fraction of values to replace with missingness in [0.0, 1.0].
    random_state : int, default=42
        Random seed for deterministic masking.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        - Perturbed feature DataFrame (defensive copy, original unmutated).
        - Advisory warnings.

    Raises
    ------
    ValueError
        If df is empty or severity is outside [0.0, 1.0].
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected df to be a pandas DataFrame, got {type(df).__name__}.")
    if df.empty or len(df) == 0:
        raise ValueError("Cannot perturb an empty DataFrame.")
    if severity is None or np.isnan(severity) or np.isinf(severity) or severity < 0.0 or severity > 1.0:
        raise ValueError(f"Severity for missingness injection must be in [0.0, 1.0]; received {severity}.")

    df_pert = df.copy(deep=True)
    warnings: List[str] = []

    if severity == 0.0:
        return df_pert, warnings

    if columns is not None:
        missing_cols = [c for c in columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Specified column(s) not found in DataFrame: {missing_cols}")
        target_cols = list(columns)
    else:
        target_cols = list(df.columns)

    if not target_cols:
        warnings.append("No feature columns available for missingness injection.")
        return df_pert, warnings

    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)

    for idx, col in enumerate(target_cols):
        col_seed = _derive_column_seed(random_state, idx, col)
        rng = np.random.RandomState(col_seed)

        mask = rng.rand(len(df)) < severity
        if col in numeric_cols:
            df_pert.loc[mask, col] = np.nan
        else:
            df_pert.loc[mask, col] = None

    return df_pert, warnings


def apply_category_dropout(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    severity: float = 0.10,
    unseen_token: str = "__UNSEEN__",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, List[str]]:
    """Inject unexpected or out-of-vocabulary categories into categorical features.

    Safely handles both object/string columns and native pandas.CategoricalDtype series.

    Parameters
    ----------
    df : pd.DataFrame
        Input tabular feature DataFrame. Unmutated defensive copy is returned.
    columns : sequence of str, optional
        Target categorical columns. If None, all non-numeric columns.
    severity : float, default=0.10
        Fraction of non-null categorical values to replace with unseen_token in [0.0, 1.0].
    unseen_token : str, default='__UNSEEN__'
        Out-of-vocabulary token used to simulate unseen categorical drift.
    random_state : int, default=42
        Random seed for deterministic corruption.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        - Perturbed feature DataFrame (defensive copy, original unmutated).
        - Advisory warnings.

    Raises
    ------
    ValueError
        If df is empty, severity outside [0.0, 1.0], or columns contain numeric features.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected df to be a pandas DataFrame, got {type(df).__name__}.")
    if df.empty or len(df) == 0:
        raise ValueError("Cannot perturb an empty DataFrame.")
    if severity is None or np.isnan(severity) or np.isinf(severity) or severity < 0.0 or severity > 1.0:
        raise ValueError(f"Severity for category dropout must be in [0.0, 1.0]; received {severity}.")

    df_pert = df.copy(deep=True)
    warnings: List[str] = []

    if severity == 0.0:
        return df_pert, warnings

    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)

    if columns is not None:
        missing_cols = [c for c in columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Specified column(s) not found in DataFrame: {missing_cols}")
        num_found = [c for c in columns if c in numeric_cols]
        if num_found:
            raise ValueError(
                f"Category dropout cannot be applied to numeric columns. Numeric columns: {num_found}"
            )
        target_cols = list(columns)
    else:
        target_cols = [c for c in df.columns if c not in numeric_cols]

    if not target_cols:
        warnings.append("No categorical feature columns available for category dropout.")
        return df_pert, warnings

    for idx, col in enumerate(target_cols):
        col_series = df[col]
        col_seed = _derive_column_seed(random_state, idx, col)
        rng = np.random.RandomState(col_seed)

        # Detect collision with existing values or declared categories
        existing_vals = set(col_series.dropna().unique())
        is_categorical = isinstance(df_pert[col].dtype, pd.CategoricalDtype)
        if is_categorical:
            existing_vals.update(df_pert[col].cat.categories)

        active_token = unseen_token
        if active_token in existing_vals:
            suffix = 1
            while f"{unseen_token}_{suffix}" in existing_vals:
                suffix += 1
            active_token = f"{unseen_token}_{suffix}"
            warnings.append(
                f"Column '{col}' already contains '{unseen_token}'; adapted to genuinely unseen token '{active_token}'."
            )

        # Only perturb observed non-null entries
        valid_mask = ~col_series.isna()
        rand_draws = rng.rand(len(df)) < severity
        mask = valid_mask & rand_draws

        # Handle CategoricalDtype safely to prevent TypeError: Cannot setitem
        if is_categorical:
            if active_token not in df_pert[col].cat.categories:
                df_pert[col] = df_pert[col].cat.add_categories([active_token])
        elif df_pert[col].dtype.kind not in ("O", "U", "S"):
            df_pert[col] = df_pert[col].astype(object)

        df_pert.loc[mask, col] = active_token

    return df_pert, warnings


def apply_compound_perturbation(
    df: pd.DataFrame,
    steps: Optional[Sequence[PerturbationConfig]] = None,
    severity: float = 0.10,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, List[str]]:
    """Apply multiple feature perturbations in a deterministic, non-mutating sequential pipeline.

    Canonical Execution Sequence (when steps is None):
    1. GAUSSIAN_NOISE: Perturbs observed numeric features with zero-mean additive noise.
    2. CATEGORY_DROPOUT: Injects unseen tokens into categorical features.
    3. MISSINGNESS_INJECTION: Replaces a fraction of feature values with null/NaN (MCAR).

    Parameters
    ----------
    df : pd.DataFrame
        Input tabular feature DataFrame. Unmutated defensive copy is returned.
    steps : sequence of PerturbationConfig, optional
        Custom ordered list of perturbation steps. If None, default canonical sequence is used.
    severity : float, default=0.10
        Magnitude parameter applied across steps in [0.0, 1.0].
    random_state : int, default=42
        Master random seed for reproducible compounding.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        - Perturbed feature DataFrame (defensive copy, original unmutated).
        - Consolidated advisory notices from all compound stages.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected df to be a pandas DataFrame, got {type(df).__name__}.")
    if df.empty or len(df) == 0:
        raise ValueError("Cannot perturb an empty DataFrame.")
    if severity is None or np.isnan(severity) or np.isinf(severity) or severity < 0.0 or severity > 1.0:
        raise ValueError(f"Severity for compound perturbation must be in [0.0, 1.0]; received {severity}.")

    df_current = df.copy(deep=True)
    all_warnings: List[str] = []

    if severity == 0.0:
        return df_current, all_warnings

    if steps is None:
        active_steps = [
            PerturbationConfig(
                perturbation_type=PerturbationType.GAUSSIAN_NOISE,
                severity_levels=[severity],
                random_state=random_state,
            ),
            PerturbationConfig(
                perturbation_type=PerturbationType.CATEGORY_DROPOUT,
                severity_levels=[severity],
                random_state=random_state,
            ),
            PerturbationConfig(
                perturbation_type=PerturbationType.MISSINGNESS_INJECTION,
                severity_levels=[severity],
                random_state=random_state,
            ),
        ]
    else:
        if len(steps) == 0:
            raise ValueError("Compound steps list must not be empty.")
        for s in steps:
            if s.perturbation_type == PerturbationType.COMPOUND:
                raise ValueError("Nested compound steps are not permitted.")
        active_steps = list(steps)

    for idx, step_cfg in enumerate(active_steps):
        # Derive independent step seed to avoid cross-stage correlation
        step_seed = (random_state + idx * 54323) % (2**31 - 1)
        p_type = step_cfg.perturbation_type
        cols = step_cfg.target_columns

        if p_type == PerturbationType.GAUSSIAN_NOISE:
            df_current, step_warn = apply_gaussian_noise(
                df=df_current, columns=cols, severity=severity, random_state=step_seed
            )
        elif p_type == PerturbationType.MISSINGNESS_INJECTION:
            df_current, step_warn = apply_missingness_injection(
                df=df_current, columns=cols, severity=severity, random_state=step_seed
            )
        elif p_type == PerturbationType.CATEGORY_DROPOUT:
            df_current, step_warn = apply_category_dropout(
                df=df_current, columns=cols, severity=severity, random_state=step_seed
            )
        else:
            raise ValueError(f"Unsupported component perturbation type: '{p_type}'.")

        for w in step_warn:
            all_warnings.append(f"[{p_type.value}]: {w}")

    return df_current, all_warnings


def perturb_dataset(
    df: pd.DataFrame,
    config: PerturbationConfig,
    severity: float,
) -> Tuple[pd.DataFrame, List[str]]:
    """Unified dispatcher executing a single perturbation transformation on a tabular dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input tabular feature DataFrame. Unmutated defensive copy is returned.
    config : PerturbationConfig
        Perturbation configuration specification.
    severity : float
        Magnitude parameter for this perturbation step.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        Perturbed DataFrame copy and diagnostic warnings.
    """
    if config is None:
        raise ValueError("PerturbationConfig must not be None.")

    p_type = config.perturbation_type
    cols = config.target_columns
    seed = config.random_state

    if p_type == PerturbationType.GAUSSIAN_NOISE:
        return apply_gaussian_noise(df=df, columns=cols, severity=severity, random_state=seed)
    elif p_type == PerturbationType.MISSINGNESS_INJECTION:
        return apply_missingness_injection(df=df, columns=cols, severity=severity, random_state=seed)
    elif p_type == PerturbationType.CATEGORY_DROPOUT:
        return apply_category_dropout(df=df, columns=cols, severity=severity, random_state=seed)
    elif p_type == PerturbationType.COMPOUND:
        return apply_compound_perturbation(
            df=df, steps=config.compound_steps, severity=severity, random_state=seed
        )
    else:
        raise ValueError(f"Unsupported perturbation type: '{p_type}'.")


def _compute_model_metric(
    estimator: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    metric_name: str,
) -> Tuple[Optional[float], Optional[str]]:
    """Compute requested classification metric using MLForge metric engine with undefined handling."""
    if metric_name not in METRIC_DIRECTIONS:
        raise ValueError(
            f"Unsupported metric '{metric_name}' for robustness evaluation. "
            f"Supported metrics: {sorted(list(METRIC_DIRECTIONS.keys()))}"
        )

    # 1. Generate predictions
    try:
        y_pred = estimator.predict(X)
    except Exception as exc:
        return None, f"Estimator predict failed: {exc}"

    # 2. Generate probabilities if probability metric
    y_proba = None
    prob_metrics = {"log_loss", "brier_score", "roc_auc"}
    if metric_name in prob_metrics:
        if not hasattr(estimator, "predict_proba"):
            return None, f"Estimator does not implement predict_proba required for '{metric_name}'."
        try:
            y_proba = estimator.predict_proba(X)
        except Exception as exc:
            return None, f"Estimator predict_proba failed: {exc}"

    # 3. Class labels
    class_labels = getattr(estimator, "classes_", np.unique(y))
    class_labels_list = list(class_labels)

    try:
        metrics_dict, _ = calculate_fold_classification_metrics(
            y_true=y,
            y_pred=y_pred,
            y_proba=y_proba,
            class_labels=class_labels_list,
        )
    except Exception as exc:
        return None, f"Metric computation error for '{metric_name}': {exc}"

    metric_val = metrics_dict.get(metric_name)
    if metric_val is None:
        return None, f"Metric '{metric_name}' was not produced by metric engine."
    if metric_val.is_undefined:
        return None, metric_val.undefined_reason or f"Metric '{metric_name}' is undefined on evaluated slice."

    return metric_val.value, None


def calculate_retention_metrics(
    clean_score: Optional[float],
    perturbed_score: Optional[float],
    metric_name: str = "accuracy",
    direction: Optional[MetricDirection] = None,
) -> Tuple[Optional[float], Optional[float], bool, Optional[str]]:
    """Calculate retained performance ratio and percentage drop with zero-denominator safeguards.

    Parameters
    ----------
    clean_score : float, optional
        Baseline performance on unperturbed data.
    perturbed_score : float, optional
        Performance under perturbation.
    metric_name : str, default='accuracy'
        Metric identifier.
    direction : MetricDirection, optional
        Directional interpretation of metric. If None, resolved via METRIC_DIRECTIONS.

    Returns
    -------
    Tuple[Optional[float], Optional[float], bool, Optional[str]]
        - retained_ratio: Ratio of retained performance/efficiency
        - performance_drop_percentage: Relative drop percentage (positive indicates degradation)
        - is_usable: True if retention metrics could be calculated
        - unusable_reason: Explanation if unusable
    """
    if clean_score is None or perturbed_score is None:
        return None, None, False, f"METRIC_UNDEFINED: Metric '{metric_name}' is undefined on clean or perturbed data."

    if np.isnan(clean_score) or np.isinf(clean_score) or np.isnan(perturbed_score) or np.isinf(perturbed_score):
        return None, None, False, f"NON_FINITE_METRIC: Non-finite score encountered for '{metric_name}'."

    metric_dir = direction or METRIC_DIRECTIONS.get(metric_name, MetricDirection.HIGHER_IS_BETTER)

    if metric_dir == MetricDirection.LOWER_IS_BETTER:
        # Loss / error metrics (e.g. log_loss, brier_score)
        if clean_score == 0.0:
            return None, None, False, f"ZERO_BASELINE_SCORE: Baseline '{metric_name}' is 0.0; relative drop cannot be computed."
        if clean_score < 0.0 or perturbed_score < 0.0:
            return None, None, False, f"NEGATIVE_SCORE_LOSS: Loss metric '{metric_name}' cannot be negative."

        drop_pct = float(round(((perturbed_score - clean_score) / clean_score) * 100.0, 6))

        if perturbed_score == 0.0:
            return None, drop_pct, False, f"ZERO_PERTURBED_LOSS: Perturbed score is 0.0; efficiency ratio undefined."

        retained_ratio = float(round(clean_score / perturbed_score, 6))
        return retained_ratio, drop_pct, True, None

    else:
        # Higher-is-better metrics (e.g. accuracy, balanced_accuracy, macro_f1, roc_auc)
        if clean_score == 0.0:
            return None, None, False, f"ZERO_BASELINE_SCORE: Baseline '{metric_name}' is 0.0; relative drop cannot be computed."
        if clean_score < 0.0:
            return None, None, False, f"NEGATIVE_BASELINE_SCORE: Baseline '{metric_name}' is negative ({clean_score}); relative drop is misleading."

        retained_ratio = float(round(perturbed_score / clean_score, 6))
        drop_pct = float(round(((clean_score - perturbed_score) / clean_score) * 100.0, 6))
        return retained_ratio, drop_pct, True, None


def calculate_aurdc(
    points: Sequence[RobustnessPerturbationResult],
) -> Tuple[Optional[float], List[str]]:
    """Compute Area Under Robustness Degradation Curve (AURDC) via normalized trapezoidal quadrature.

    Parameters
    ----------
    points : sequence of RobustnessPerturbationResult
        Ordered list of evaluated perturbation results across severity levels.

    Returns
    -------
    Tuple[Optional[float], List[str]]
        - aurdc: Normalized area under retention curve, or None if incomplete/unusable.
        - warnings: Explanatory warnings if AURDC cannot be computed.
    """
    warnings: List[str] = []
    if not points:
        warnings.append("Cannot calculate AURDC for empty points list.")
        return None, warnings

    has_unusable = any(
        not p.is_usable
        or p.retained_ratio is None
        or not math.isfinite(p.retained_ratio)
        or not math.isfinite(p.severity_level)
        for p in points
    )
    if has_unusable:
        warnings.append(
            "Area under degradation curve is undefined because one or more perturbation points failed or are unusable."
        )
        return None, warnings

    sorted_points = sorted(points, key=lambda p: p.severity_level)
    sevs = [p.severity_level for p in sorted_points]
    if len(set(sevs)) != len(sevs):
        warnings.append(
            "Cannot calculate AURDC: duplicate severity levels detected in perturbation points."
        )
        return None, warnings

    if len(sorted_points) == 1:
        return round(float(sorted_points[0].retained_ratio), 6), warnings

    retentions = [p.retained_ratio for p in sorted_points]
    max_sev = sevs[-1]
    min_sev = sevs[0]

    if max_sev > min_sev:
        trapezoid_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        if trapezoid_fn is not None:
            integral = float(trapezoid_fn(retentions, sevs))
        else:
            y = np.asarray(retentions, dtype=float)
            x = np.asarray(sevs, dtype=float)
            integral = float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))
        aurdc = round(integral / (max_sev - min_sev), 6)
        return aurdc, warnings
    else:
        return round(float(retentions[0]), 6), warnings


def evaluate_perturbation_step(
    estimator: Any,
    X_val: pd.DataFrame,
    y_val: Union[pd.Series, np.ndarray, Sequence[Any]],
    config: PerturbationConfig,
    severity: float,
    clean_score: Optional[float] = None,
    metric_name: str = "accuracy",
) -> RobustnessPerturbationResult:
    """Evaluate a fitted estimator under a single perturbation severity level without refitting.

    Parameters
    ----------
    estimator : Any
        Fitted estimator or scikit-learn Pipeline. Must implement .predict(X) or .predict_proba(X).
    X_val : pd.DataFrame
        Unperturbed validation feature partition.
    y_val : array-like
        Ground-truth validation target labels.
    config : PerturbationConfig
        Perturbation configuration.
    severity : float
        Severity level to evaluate.
    clean_score : float, optional
        Baseline score on clean data. If None, calculated directly.
    metric_name : str, default='accuracy'
        Target metric evaluated (e.g. 'accuracy', 'balanced_accuracy', 'macro_f1', 'log_loss', 'brier_score').

    Returns
    -------
    RobustnessPerturbationResult
        Structured point evaluation result under perturbation.
    """
    if estimator is None:
        raise ValueError("Estimator must not be None.")
    if X_val is None or len(X_val) == 0:
        raise ValueError("X_val must be a non-empty DataFrame.")
    if y_val is None or len(y_val) != len(X_val):
        raise ValueError("y_val must match X_val length.")

    y_val_arr = np.asarray(y_val)

    # 1. Compute clean baseline score if not provided
    if clean_score is None:
        c_score, c_err = _compute_model_metric(
            estimator=estimator,
            X=X_val,
            y=y_val_arr,
            metric_name=metric_name,
        )
        if c_score is None:
            return RobustnessPerturbationResult(
                perturbation_type=config.perturbation_type,
                severity_level=round(severity, 6),
                metric_name=metric_name,
                is_usable=False,
                unusable_reason=f"Baseline clean metric evaluation failed: {c_err}",
            )
        clean_score = c_score

    # 2. Perturb validation features defensively
    try:
        X_pert, pert_warnings = perturb_dataset(df=X_val, config=config, severity=severity)
    except Exception as exc:
        return RobustnessPerturbationResult(
            perturbation_type=config.perturbation_type,
            severity_level=round(severity, 6),
            metric_name=metric_name,
            clean_score=clean_score,
            is_usable=False,
            unusable_reason=f"Perturbation transformation failed: {exc}",
        )

    # 3. Predict on perturbed validation partition without refitting and evaluate metric
    p_score, p_err = _compute_model_metric(
        estimator=estimator,
        X=X_pert,
        y=y_val_arr,
        metric_name=metric_name,
    )
    if p_score is None:
        return RobustnessPerturbationResult(
            perturbation_type=config.perturbation_type,
            severity_level=round(severity, 6),
            metric_name=metric_name,
            clean_score=clean_score,
            is_usable=False,
            unusable_reason=f"Estimator prediction failed under perturbation: {p_err}",
            warnings=pert_warnings,
        )

    # 4. Compute relative retention and drop metrics
    retained_ratio, drop_pct, usable, unusable_reason = calculate_retention_metrics(
        clean_score=clean_score,
        perturbed_score=p_score,
        metric_name=metric_name,
    )

    return RobustnessPerturbationResult(
        perturbation_type=config.perturbation_type,
        severity_level=round(severity, 6),
        metric_name=metric_name,
        clean_score=clean_score,
        perturbed_score=p_score,
        retained_ratio=retained_ratio,
        performance_drop_percentage=drop_pct,
        is_usable=usable,
        unusable_reason=unusable_reason,
        warnings=pert_warnings,
    )


def evaluate_robustness_curve(
    estimator: Any,
    X_val: pd.DataFrame,
    y_val: Union[pd.Series, np.ndarray, Sequence[Any]],
    config: PerturbationConfig,
    metric_name: str = "accuracy",
    clean_score: Optional[float] = None,
) -> RobustnessCurve:
    """Evaluate performance degradation trajectory across incremental perturbation severities.

    Parameters
    ----------
    estimator : Any
        Fitted estimator or Pipeline.
    X_val : pd.DataFrame
        Unperturbed validation features.
    y_val : array-like
        Validation targets.
    config : PerturbationConfig
        Perturbation configuration with severity_levels.
    metric_name : str, default='accuracy'
        Metric evaluated.
    clean_score : float, optional
        Pre-calculated baseline score.

    Returns
    -------
    RobustnessCurve
        Degradation curve across tested severities with computed AURDC.
    """
    if config is None:
        raise ValueError("PerturbationConfig must not be None.")

    y_val_arr = np.asarray(y_val)

    if clean_score is None:
        c_score, c_err = _compute_model_metric(
            estimator=estimator,
            X=X_val,
            y=y_val_arr,
            metric_name=metric_name,
        )
        if c_score is None:
            return RobustnessCurve(
                perturbation_type=config.perturbation_type,
                metric_name=metric_name,
                clean_score=None,
                points=[
                    RobustnessPerturbationResult(
                        perturbation_type=config.perturbation_type,
                        severity_level=round(sev, 6),
                        metric_name=metric_name,
                        is_usable=False,
                        unusable_reason=f"Baseline clean metric evaluation failed: {c_err}",
                    )
                    for sev in config.severity_levels
                ],
                area_under_degradation_curve=None,
                warnings=[f"Baseline clean prediction failed for '{metric_name}': {c_err}"],
            )
        clean_score = c_score

    points: List[RobustnessPerturbationResult] = []
    curve_warnings: List[str] = []

    for sev in config.severity_levels:
        pt = evaluate_perturbation_step(
            estimator=estimator,
            X_val=X_val,
            y_val=y_val,
            config=config,
            severity=sev,
            clean_score=clean_score,
            metric_name=metric_name,
        )
        points.append(pt)
        if pt.warnings:
            curve_warnings.extend(pt.warnings)

    # Compute Area Under Robustness Degradation Curve (AURDC)
    auc_degradation, aurdc_warnings = calculate_aurdc(points)
    if aurdc_warnings:
        curve_warnings.extend(aurdc_warnings)

    return RobustnessCurve(
        perturbation_type=config.perturbation_type,
        metric_name=metric_name,
        clean_score=clean_score,
        points=points,
        area_under_degradation_curve=auc_degradation,
        warnings=sorted(list(set(curve_warnings))),
    )


def evaluate_model_robustness(
    estimator: Any,
    X_val: pd.DataFrame,
    y_val: Union[pd.Series, np.ndarray, Sequence[Any]],
    configs: Sequence[PerturbationConfig],
    metric_name: str = "accuracy",
) -> RobustnessEvaluationResult:
    """Evaluate model robustness across multiple perturbation dimensions.

    Parameters
    ----------
    estimator : Any
        Fitted model or Pipeline.
    X_val : pd.DataFrame
        Unperturbed validation features.
    y_val : array-like
        Validation targets.
    configs : sequence of PerturbationConfig
        List of perturbation configurations to evaluate.
    metric_name : str, default='accuracy'
        Metric evaluated.

    Returns
    -------
    RobustnessEvaluationResult
        Aggregated robustness evaluation result including per-curve and overall AURDC.
    """
    if not configs:
        return RobustnessEvaluationResult(
            curves=[],
            summary_retention={},
            warnings=["No perturbation configurations provided."],
        )

    curves: List[RobustnessCurve] = []
    summary_retention: Dict[str, float] = {}
    all_warnings: List[str] = []

    y_val_arr = np.asarray(y_val)
    clean_score, clean_err = _compute_model_metric(
        estimator=estimator,
        X=X_val,
        y=y_val_arr,
        metric_name=metric_name,
    )

    if clean_score is None:
        all_warnings.append(f"Clean baseline evaluation failed for '{metric_name}': {clean_err}")

    for cfg in configs:
        curve = evaluate_robustness_curve(
            estimator=estimator,
            X_val=X_val,
            y_val=y_val,
            config=cfg,
            metric_name=metric_name,
            clean_score=clean_score,
        )
        curves.append(curve)
        if curve.warnings:
            all_warnings.extend(curve.warnings)

        if curve.points:
            max_pt = max(curve.points, key=lambda p: p.severity_level)
            if max_pt.is_usable and max_pt.retained_ratio is not None:
                summary_retention[f"{cfg.perturbation_type.value}_max_retention"] = max_pt.retained_ratio
                if max_pt.performance_drop_percentage is not None:
                    summary_retention[f"{cfg.perturbation_type.value}_max_drop_pct"] = max_pt.performance_drop_percentage
            else:
                all_warnings.append(
                    f"Perturbation '{cfg.perturbation_type.value}' failed at maximum severity ({max_pt.severity_level}); summary retention unavailable."
                )

        if curve.area_under_degradation_curve is not None:
            summary_retention[f"{cfg.perturbation_type.value}_aurdc"] = curve.area_under_degradation_curve

    all_curves_usable = bool(curves) and all(
        c.points and all(p.is_usable for p in c.points) for c in curves
    )
    if not all_curves_usable:
        all_warnings.append("One or more perturbation curves contain unusable points or evaluation failures.")

    valid_aurdcs = [c.area_under_degradation_curve for c in curves if c.area_under_degradation_curve is not None]
    if valid_aurdcs and len(valid_aurdcs) == len(curves):
        summary_retention["mean_aurdc"] = round(float(np.mean(valid_aurdcs)), 6)

    return RobustnessEvaluationResult(
        curves=curves,
        summary_retention=summary_retention,
        is_usable=all_curves_usable,
        warnings=sorted(list(set(all_warnings))),
    )


# ============================================================================
# Pre-Packaged Stress Testing Profiles (Milestone 3.2)
# ============================================================================

def create_noise_stress_profile(
    target_columns: Optional[Sequence[str]] = None,
    severity_levels: Optional[Sequence[float]] = None,
    random_state: int = 42,
) -> PerturbationConfig:
    """Create standard Gaussian noise stress testing profile (default grid: 5%, 10%, 25%, 50%)."""
    return PerturbationConfig(
        perturbation_type=PerturbationType.GAUSSIAN_NOISE,
        severity_levels=list(severity_levels) if severity_levels is not None else list(STANDARD_SEVERITY_GRID),
        target_columns=list(target_columns) if target_columns is not None else None,
        random_state=random_state,
    )


def create_missingness_stress_profile(
    target_columns: Optional[Sequence[str]] = None,
    severity_levels: Optional[Sequence[float]] = None,
    random_state: int = 42,
) -> PerturbationConfig:
    """Create standard MCAR missingness stress testing profile (default grid: 5%, 10%, 25%, 50%)."""
    return PerturbationConfig(
        perturbation_type=PerturbationType.MISSINGNESS_INJECTION,
        severity_levels=list(severity_levels) if severity_levels is not None else list(STANDARD_SEVERITY_GRID),
        target_columns=list(target_columns) if target_columns is not None else None,
        random_state=random_state,
    )


def create_category_dropout_stress_profile(
    target_columns: Optional[Sequence[str]] = None,
    severity_levels: Optional[Sequence[float]] = None,
    random_state: int = 42,
) -> PerturbationConfig:
    """Create standard category dropout stress testing profile (default grid: 5%, 10%, 25%, 50%)."""
    return PerturbationConfig(
        perturbation_type=PerturbationType.CATEGORY_DROPOUT,
        severity_levels=list(severity_levels) if severity_levels is not None else list(STANDARD_SEVERITY_GRID),
        target_columns=list(target_columns) if target_columns is not None else None,
        random_state=random_state,
    )


def create_compound_stress_profile(
    target_columns: Optional[Sequence[str]] = None,
    severity_levels: Optional[Sequence[float]] = None,
    compound_steps: Optional[Sequence[PerturbationConfig]] = None,
    random_state: int = 42,
) -> PerturbationConfig:
    """Create standard multi-feature compound stress testing profile."""
    return PerturbationConfig(
        perturbation_type=PerturbationType.COMPOUND,
        severity_levels=list(severity_levels) if severity_levels is not None else list(STANDARD_SEVERITY_GRID),
        target_columns=list(target_columns) if target_columns is not None else None,
        random_state=random_state,
        compound_steps=list(compound_steps) if compound_steps is not None else None,
    )


def get_standard_stress_suite(random_state: int = 42) -> List[PerturbationConfig]:
    """Return the complete suite of standard stress testing profiles."""
    return [
        create_noise_stress_profile(random_state=random_state),
        create_missingness_stress_profile(random_state=random_state),
        create_category_dropout_stress_profile(random_state=random_state),
        create_compound_stress_profile(random_state=random_state),
    ]


def evaluate_stress_profile(
    estimator: Any,
    X_val: pd.DataFrame,
    y_val: Union[pd.Series, np.ndarray, Sequence[Any]],
    profile: Optional[Union[PerturbationConfig, Sequence[PerturbationConfig]]] = None,
    metric_name: str = "accuracy",
) -> RobustnessEvaluationResult:
    """Evaluate an estimator against a pre-packaged stress profile or the full standard suite."""
    if profile is None:
        configs = get_standard_stress_suite()
    elif isinstance(profile, PerturbationConfig):
        configs = [profile]
    else:
        configs = list(profile)

    return evaluate_model_robustness(
        estimator=estimator,
        X_val=X_val,
        y_val=y_val,
        configs=configs,
        metric_name=metric_name,
    )
