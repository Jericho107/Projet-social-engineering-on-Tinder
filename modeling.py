from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


def build_preprocessor(df: pd.DataFrame, feature_columns: Iterable[str]) -> tuple[ColumnTransformer, list[str], list[str]]:
    selected = [column for column in feature_columns if column in df.columns]
    numeric_features = [column for column in selected if pd.api.types.is_numeric_dtype(df[column])]
    categorical_features = [column for column in selected if column not in numeric_features]
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_features, categorical_features


def candidate_models(random_state: int = 42) -> dict[str, object]:
    models: dict[str, object] = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": None,
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        models["XGBoost"] = GradientBoostingClassifier(random_state=random_state)
    return models


def evaluate_models(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    target_column: str = "match",
    test_size: float = 0.2,
    random_state: int = 42,
    cv_splits: int = 5,
) -> dict:
    working = df.copy()
    working = working.dropna(subset=[target_column])
    feature_columns = [column for column in feature_columns if column in working.columns]
    X = working[feature_columns]
    y = working[target_column].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    preprocessor, numeric_features, categorical_features = build_preprocessor(working, feature_columns)
    results = []
    roc_data = {}
    confusion_matrices = {}
    fitted_models = {}
    models = candidate_models(random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    splitter = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    for model_name, estimator in models.items():
        if estimator is None:
            continue
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        cv_scores = {}
        for score_name, scorer in scoring.items():
            cv_scores[score_name] = cross_val_score(pipeline, X_train, y_train, cv=splitter, scoring=scorer, n_jobs=-1).mean()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            decision = pipeline.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-decision))
        metrics = {
            "model": model_name,
            **cv_scores,
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred, zero_division=0),
            "test_recall": recall_score(y_test, y_pred, zero_division=0),
            "test_f1": f1_score(y_test, y_pred, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, y_prob),
        }
        results.append(metrics)
        confusion_matrices[model_name] = confusion_matrix(y_test, y_pred).tolist()
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_data[model_name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        }
        fitted_models[model_name] = pipeline
    metrics_frame = pd.DataFrame(results).sort_values("test_roc_auc", ascending=False)
    best_model_name = metrics_frame.iloc[0]["model"] if not metrics_frame.empty else None
    best_model = fitted_models.get(best_model_name)
    return {
        "feature_columns": list(feature_columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "metrics": metrics_frame,
        "confusion_matrices": confusion_matrices,
        "roc_data": roc_data,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
    }


def shap_feature_frame(df: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    selected = [column for column in feature_columns if column in df.columns]
    frame = df[selected].copy()
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = pd.Categorical(frame[column]).codes.astype(float)
    return frame.fillna(frame.median(numeric_only=True))


def leakage_comparison(
    df: pd.DataFrame,
    realistic_features: Iterable[str],
    leaky_features: Iterable[str],
    target_column: str = "match",
    random_state: int = 42,
) -> dict:
    realistic = evaluate_models(df, realistic_features, target_column=target_column, random_state=random_state)
    leaky = evaluate_models(df, leaky_features, target_column=target_column, random_state=random_state)
    return {
        "realistic": {
            "best_model_name": realistic["best_model_name"],
            "metrics": realistic["metrics"].to_dict(orient="records"),
        },
        "leaky": {
            "best_model_name": leaky["best_model_name"],
            "metrics": leaky["metrics"].to_dict(orient="records"),
        },
    }


def save_bundle(path: Path, bundle: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: Path) -> dict:
    return joblib.load(path)


def predict_with_bundle(bundle: dict, input_frame: pd.DataFrame) -> np.ndarray:
    model = bundle["best_model"]
    if model is None:
        raise ValueError("No trained model available in the provided bundle.")
    probabilities = model.predict_proba(input_frame)[:, 1]
    return probabilities
