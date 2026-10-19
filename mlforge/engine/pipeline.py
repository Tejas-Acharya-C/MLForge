"""MLForge Fold-Local Pipeline Construction.

Constructs scikit-learn preprocessing pipelines encapsulated with candidate or
baseline estimators, ensuring zero data leakage across cross-validation folds.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlforge.core.exceptions import ConfigurationError


def build_fold_pipeline(
    model_type: str,
    numeric_features: List[str],
    categorical_features: List[str],
    hyperparameters: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Pipeline:
    """Build a fresh, unfitted scikit-learn Pipeline with fold-local preprocessing.

    Parameters
    ----------
    model_type : str
        Either 'dummy_classifier' or 'logistic_regression'.
    numeric_features : List[str]
        List of numeric feature column names.
    categorical_features : List[str]
        List of categorical feature column names.
    hyperparameters : Optional[Dict[str, Any]]
        Optional hyperparameters for candidate model.
    random_state : int, default=42
        Seed for reproducible estimator initialization.

    Returns
    -------
    Pipeline
        Unfitted Pipeline containing ColumnTransformer and estimator.

    Raises
    ------
    ConfigurationError
        If model_type is unsupported or no features are provided.
    """
    hyperparams = hyperparameters.copy() if hyperparameters else {}
    transformers = []

    # 1. Numeric Preprocessing: Median Imputation + Standardization
    if numeric_features:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    # 2. Categorical Preprocessing: Most Frequent Imputation + One-Hot Encoding
    if categorical_features:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    if not transformers:
        raise ConfigurationError("Cannot build pipeline with zero numeric and zero categorical features.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # 3. Estimator Selection
    if model_type == "dummy_classifier":
        strategy = hyperparams.pop("strategy", "prior")
        estimator = DummyClassifier(strategy=strategy, random_state=random_state)
    elif model_type == "logistic_regression":
        solver = hyperparams.pop("solver", "lbfgs")
        max_iter = hyperparams.pop("max_iter", 1000)
        c_param = hyperparams.pop("C", 1.0)
        penalty = hyperparams.pop("penalty", "l2")
        estimator = LogisticRegression(
            solver=solver,
            max_iter=max_iter,
            C=c_param,
            penalty=penalty,
            random_state=random_state,
            **hyperparams,
        )
    else:
        raise ConfigurationError(
            f"Unsupported model_type: '{model_type}'. Expected 'dummy_classifier' or 'logistic_regression'."
        )

    # 4. Encapsulate into Pipeline ensuring preprocessor is fitted strictly with estimator
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator),
    ])
