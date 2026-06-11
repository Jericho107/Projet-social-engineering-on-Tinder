from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "final_dataset.csv"
MODEL_PATH = ROOT / "reports" / "model_bundle.joblib"


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(PROCESSED)


@st.cache_resource
def load_model() -> dict:
    return joblib.load(MODEL_PATH)


def build_prediction_frame(bundle: dict, age: float, age_o: float, attr: float, fun: float, intel: float, like: float, interest_alignment: float) -> pd.DataFrame:
    feature_columns = bundle.get("feature_columns", [])
    values = {column: 0 for column in feature_columns}
    values.update(
        {
            "age": age,
            "age_o": age_o,
            "attr": attr,
            "fun": fun,
            "intel": intel,
            "like": like,
            "age_gap": abs(age - age_o),
            "preference_alignment": interest_alignment / 10.0,
            "compatibility_score": (interest_alignment / 10.0 + attr / 10.0 + like / 10.0) / 3.0,
            "personality_score": (fun + intel) / 2.0,
            "attractiveness_weight": attr,
            "samerace": 0,
            "int_corr": 0,
        }
    )
    frame = pd.DataFrame([values])
    for column in feature_columns:
        if column not in frame.columns:
            frame[column] = 0
    return frame[feature_columns]


st.set_page_config(page_title="Speed Dating Analytics", layout="wide")
st.title("Speed Dating Analytics Dashboard")

data = load_data()
bundle = load_model()

page = st.sidebar.radio("Navigation", ["Overview", "Behaviour Analysis", "Statistical Insights", "Match Prediction"])

if page == "Overview":
    col1, col2, col3 = st.columns(3)
    col1.metric("Match Rate", f"{data['match'].mean():.1%}")
    col2.metric("Yes Rate", f"{data['dec'].mean():.1%}")
    col3.metric("Participants", f"{data['iid'].nunique():,}")
    st.dataframe(data[["age", "age_o", "attr", "fun", "like", "match"]].head(20))
elif page == "Behaviour Analysis":
    st.subheader("Behaviour signals")
    st.line_chart(data[["attr", "fun", "intel", "like"]].dropna().head(250))
    st.dataframe(data[["age", "age_o", "samerace", "match", "dec"]].describe().T)
elif page == "Statistical Insights":
    st.subheader("Statistical results")
    st.write("Use the statistical notebook and exported reports for detailed p-values and effect sizes.")
    st.write("Key variables: attractiveness, fun, intelligence, shared interests, age gap, same race.")
else:
    st.subheader("Match probability estimator")
    age = st.number_input("Age", min_value=18, max_value=60, value=28)
    age_o = st.number_input("Partner age", min_value=18, max_value=60, value=27)
    attr = st.slider("Attractiveness", 1.0, 10.0, 7.0, 0.5)
    fun = st.slider("Fun", 1.0, 10.0, 7.0, 0.5)
    intel = st.slider("Intelligence", 1.0, 10.0, 7.0, 0.5)
    like = st.slider("Like", 1.0, 10.0, 7.0, 0.5)
    importance = st.slider("Interest alignment", 1.0, 10.0, 7.0, 0.5)
    if st.button("Predict"):
        frame = build_prediction_frame(bundle, age, age_o, attr, fun, intel, like, importance)
        model = bundle["best_model"]
        probability = model.predict_proba(frame)[:, 1][0]
        st.success(f"Estimated match probability: {probability:.1%}")
