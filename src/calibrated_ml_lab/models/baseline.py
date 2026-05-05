"""Baseline supervised models used by the workflow."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class FeatureSpec:
    """Feature-column specification."""

    numeric_features: list[str]
    categorical_features: list[str]


def infer_feature_spec(frame: pd.DataFrame, target: str) -> FeatureSpec:
    """Infer numeric and categorical feature columns from a dataframe."""

    if target not in frame.columns:
        raise ValueError(f"target column '{target}' not found")
    features = [col for col in frame.columns if col != target and not col.startswith("true_")]
    numeric = [col for col in features if pd.api.types.is_numeric_dtype(frame[col])]
    categorical = [col for col in features if col not in numeric]
    return FeatureSpec(numeric_features=numeric, categorical_features=categorical)


def build_classification_pipeline(spec: FeatureSpec) -> Pipeline:
    """Build a probabilistic classifier pipeline."""

    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, spec.numeric_features),
            ("cat", categorical_pipeline, spec.categorical_features),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=7,
                    min_samples_leaf=8,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_regression_pipeline(feature_columns: list[str]) -> Pipeline:
    """Build a regression pipeline for conformal intervals."""

    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    preprocessor = ColumnTransformer([("num", numeric_pipeline, feature_columns)])
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=120,
                    max_depth=8,
                    min_samples_leaf=6,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
