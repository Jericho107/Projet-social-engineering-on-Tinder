from __future__ import annotations

import numpy as np
import pandas as pd

DECLARED_PREFERENCE_COLUMNS = ["attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"]
RATING_COLUMNS = ["attr", "sinc", "intel", "fun", "amb", "shar"]


def create_age_gap(df: pd.DataFrame) -> pd.Series:
    if {"age", "age_o"}.issubset(df.columns):
        return (df["age"] - df["age_o"]).abs()
    return pd.Series(np.nan, index=df.index)


def create_preference_alignment(df: pd.DataFrame) -> pd.Series:
    declared = [column for column in DECLARED_PREFERENCE_COLUMNS if column in df.columns]
    observed = [column for column in RATING_COLUMNS if column in df.columns]
    if not declared or not observed:
        return pd.Series(np.nan, index=df.index)
    common_length = min(len(declared), len(observed))
    declared_frame = df[declared[:common_length]]
    observed_frame = df[observed[:common_length]]
    alignment_gap = (declared_frame.subtract(observed_frame)).abs().mean(axis=1)
    return 1 - (alignment_gap / 9.0)


def create_personality_score(df: pd.DataFrame) -> pd.Series:
    personality_columns = [column for column in ["sinc", "intel", "fun", "amb", "shar"] if column in df.columns]
    if not personality_columns:
        return pd.Series(np.nan, index=df.index)
    return df[personality_columns].mean(axis=1, skipna=True)


def create_attractiveness_weight(df: pd.DataFrame) -> pd.Series:
    if {"attr", "attr1_1"}.issubset(df.columns):
        return (df["attr"].fillna(0) * df["attr1_1"].fillna(0)) / 10.0
    if "attr" in df.columns:
        return df["attr"]
    return pd.Series(np.nan, index=df.index)


def create_compatibility_score(df: pd.DataFrame) -> pd.Series:
    components = pd.DataFrame(index=df.index)
    if "age_gap" in df.columns:
        components["age_fit"] = 1 - (df["age_gap"].clip(lower=0, upper=15) / 15.0)
    if "samerace" in df.columns:
        components["same_race"] = df["samerace"].fillna(0)
    if "int_corr" in df.columns:
        components["interest_corr"] = df["int_corr"].clip(-1, 1).fillna(0).add(1).div(2)
    if "preference_alignment" in df.columns:
        components["alignment"] = df["preference_alignment"].clip(0, 1).fillna(0)
    if "like" in df.columns:
        median_like = df["like"].median() if df["like"].notna().any() else 0
        components["like_signal"] = df["like"].fillna(median_like) / 10.0
    if components.empty:
        return pd.Series(np.nan, index=df.index)
    weights = {
        "age_fit": 0.15,
        "same_race": 0.10,
        "interest_corr": 0.20,
        "alignment": 0.30,
        "like_signal": 0.25,
    }
    weighted = sum(components[column].fillna(0) * weights.get(column, 0) for column in components.columns)
    return weighted / sum(weights.get(column, 0) for column in components.columns)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    engineered["age_gap"] = create_age_gap(engineered)
    engineered["preference_alignment"] = create_preference_alignment(engineered)
    engineered["personality_score"] = create_personality_score(engineered)
    engineered["attractiveness_weight"] = create_attractiveness_weight(engineered)
    engineered["compatibility_score"] = create_compatibility_score(engineered)
    return engineered


def feature_dictionary() -> dict[str, str]:
    return {
        "age_gap": "Absolute age difference between participant and partner.",
        "preference_alignment": "Distance between declared preferences and actual partner ratings, scaled to 0-1.",
        "compatibility_score": "Composite score blending age proximity, same-race signal, interest correlation and attraction.",
        "personality_score": "Average of partner-rated sincerity, intelligence, fun, ambition and shared interests.",
        "attractiveness_weight": "Interaction between the participant's attractiveness rating and the declared attractiveness importance.",
    }
