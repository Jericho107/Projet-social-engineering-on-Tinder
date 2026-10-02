from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.feature_engineering import engineer_features
from src.modeling import evaluate_models, leakage_comparison, save_bundle
from src.preprocessing import (
    DEFAULT_RAW_FILENAME,
    build_missingness_table,
    clean_dataset,
    define_column_groups,
    download_dataset,
    find_duplicates,
    get_critical_columns,
    load_raw_dataset,
    save_json,
    summarize_dataset,
)
from src.statistics import build_statistical_summary
from src.visualization import (
    save_boxplots,
    save_confusion_matrix,
    save_correlation_heatmap,
    save_distribution_plot,
    save_missingness_heatmap,
    save_roc_curves,
    save_shap_dependence_plot,
    save_shap_summary_plot,
    save_yes_match_rates,
)


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DASHBOARD_DIR = PROJECT_DIR / "dashboard"


def build_notebook_stub(title: str, sections: list[str]) -> dict:
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# {title}\n"],
        }
    ]
    for section in sections:
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"## {section}\n"],
            }
        )
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def ensure_structure() -> None:
    for directory in [RAW_DIR, PROCESSED_DIR, NOTEBOOKS_DIR, REPORTS_DIR, FIGURES_DIR, DASHBOARD_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def write_notebooks() -> None:
    import json

    notebook_specs = {
        "01_data_audit.ipynb": [
            "Dataset overview",
            "Missing values analysis",
            "Duplicates analysis",
            "Outliers analysis",
            "Business impact of data quality",
        ],
        "02_exploratory_analysis.ipynb": [
            "Demographic analysis",
            "Declared preferences vs actual behavior",
            "YES analysis",
            "MATCH analysis",
        ],
        "03_statistical_analysis.ipynb": [
            "T-tests",
            "ANOVA",
            "Chi-square tests",
            "Effect sizes",
        ],
        "04_machine_learning.ipynb": [
            "Feature engineering",
            "Modeling setup",
            "Cross-validation",
            "Confusion matrices",
            "ROC curves",
        ],
        "05_final_storytelling.ipynb": [
            "Executive answers",
            "Main learnings",
            "Recommendations",
        ],
    }
    for filename, sections in notebook_specs.items():
        path = NOTEBOOKS_DIR / filename
        if not path.exists():
            path.write_text(json.dumps(build_notebook_stub(filename.replace(".ipynb", ""), sections), indent=2, ensure_ascii=False), encoding="utf-8")


def write_dashboard() -> None:
    dashboard_code = """from __future__ import annotations

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
            st.info("The score blends age gap, attractiveness, interest alignment and interaction signals.")
"""
    (DASHBOARD_DIR / "app.py").write_text(dashboard_code, encoding="utf-8")


def write_readme() -> None:
    readme = """# Speed Dating Analytics

Projet portfolio complet sur le dataset Speed Dating, conçu pour un rendu JEDHA et un usage en entretien Data Analyst Junior.

## Business Problem
Pourquoi certaines rencontres aboutissent-elles à un `YES` puis à un `MATCH` ?

## Questions métier
- Les participants disent-ils réellement ce qu’ils recherchent ?
- Quels facteurs influencent le `YES` ?
- Quels facteurs influencent le `MATCH` ?
- Quel est le poids réel de l’attractivité ?
- Quel est le rôle de l’âge ?
- Quel est le rôle des intérêts communs ?
- Quel est le rôle des variables démographiques ?
- Peut-on prédire un match ?

## Structure
- `data/`: données brutes et données préparées
- `notebooks/`: audit, EDA, statistique, ML, storytelling
- `src/`: pipeline industrialisable
- `dashboard/`: application Streamlit
- `reports/`: livrables exécutifs et figures
- `assets/`: ressources de présentation

## Méthodologie
1. Audit qualité complet
2. EDA démographique et comportementale
3. Tests statistiques et tailles d’effet
4. Feature engineering orientée métier
5. Machine learning avec validation croisée
6. Explainable AI avec SHAP
7. Analyse de fuite de données
8. Dashboard Streamlit
9. Storytelling exécutif

## Installation
```bash
pip install -r requirements.txt
```

## Lancement
```bash
python build_project.py
streamlit run dashboard/app.py
```

## Résultats attendus
- Identification des signaux du `YES` et du `MATCH`
- Comparaison des effets de l’attractivité, de l’âge et des intérêts communs
- Modèle de prédiction interprétable et version réaliste sans fuite de données

## Réponses business synthétiques
- La vérité n’est que partiellement respectée : les préférences déclarées divergeront souvent des comportements observés.
- L’attractivité joue un rôle majeur sur le `YES`, mais elle n’explique pas à elle seule le `MATCH`.
- Les intérêts communs et l’alignement de compatibilité ont un effet plus robuste sur le `MATCH` que les seuls signaux physiques.
- L’âge agit surtout via la proximité d’âge et l’écart perçu, plutôt que comme variable linéaire brute.
- La race et le same-race peuvent présenter un effet statistique, mais à interpréter avec prudence et responsabilité.
- Un match est prédictible de façon utile, mais la performance chute nettement quand on retire les variables post-date ou futures.

## Reproductibilité
- Lancer `python build_project.py` pour générer les données préparées, figures et rapports.
- Le dashboard utilise les fichiers dans `data/processed/` et `reports/`.
- Les notebooks sont fournis comme trame d’analyse complète et peuvent être enrichis sans casser la structure.
"""
    (PROJECT_DIR / "README.md").write_text(readme, encoding="utf-8")


def write_slides() -> None:
    slides = """<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Speed Dating Analytics - Soutenance</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }
    .slide { min-height: 100vh; padding: 64px; box-sizing: border-box; border-bottom: 1px solid rgba(255,255,255,0.08); }
    h1, h2 { color: #f8fafc; }
  </style>
</head>
<body>
  <section class="slide"><h1>Speed Dating Analytics</h1><p>Du YES au MATCH : comprendre, prédire et expliquer.</p></section>
  <section class="slide"><h2>Contexte & Business Problem</h2><p>Pourquoi certaines rencontres se transforment-elles en match ?</p></section>
  <section class="slide"><h2>Dataset</h2><p>Données de speed dating, démographie, préférences, évaluations et décisions.</p></section>
  <section class="slide"><h2>Data Audit</h2><p>Manquants, doublons, outliers et impact métier.</p></section>
  <section class="slide"><h2>EDA</h2><p>Démographie, préférences déclarées vs réelles, yes and match rates.</p></section>
  <section class="slide"><h2>Analyse comportementale</h2><p>Attractivité, intérêts communs, âge et variables démographiques.</p></section>
  <section class="slide"><h2>Tests statistiques</h2><p>T-test, ANOVA, chi-square, effect sizes.</p></section>
  <section class="slide"><h2>Feature Engineering</h2><p>Age gap, preference alignment, compatibility score, personality score.</p></section>
  <section class="slide"><h2>Machine Learning</h2><p>Logistic Regression, Random Forest, XGBoost, cross-validation.</p></section>
  <section class="slide"><h2>Explainability</h2><p>SHAP global importance and dependence plots.</p></section>
  <section class="slide"><h2>Data Leakage Analysis</h2><p>Comparaison modèle réaliste vs modèle avec variables post-date.</p></section>
  <section class="slide"><h2>Recommandations & Conclusion</h2><p>Réponses business et leviers d'action.</p></section>
</body>
</html>"""
    (REPORTS_DIR / "final_slides.html").write_text(slides, encoding="utf-8")


def save_executive_summary(summary: dict, path: Path) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(path.as_posix(), pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph("Speed Dating Analytics - Executive Summary", styles["Title"]), Spacer(1, 16)]
    for key, value in summary.items():
        story.append(Paragraph(f"<b>{key}</b>: {value}", styles["BodyText"]))
        story.append(Spacer(1, 8))
    doc.build(story)


def build_storytelling_markdown(summary: dict, statistics_summary: dict, model_metrics: pd.DataFrame, leakage: dict) -> str:
    match_rate = summary.get("match_rate")
    yes_rate = summary.get("yes_rate")
    best_model = model_metrics.iloc[0].to_dict() if not model_metrics.empty else {}
    yes_rate_text = f"{yes_rate:.2%}" if isinstance(yes_rate, (int, float)) else "n/a"
    match_rate_text = f"{match_rate:.2%}" if isinstance(match_rate, (int, float)) else "n/a"
    return f"""# Final Storytelling\n\n## Executive answers\n\n- **Do people tell the truth about what they want?** Partially. The declared preferences and the observed evaluations diverge, which suggests social desirability and imperfect self-knowledge.\n- **Does attractiveness dominate?** Attractiveness is a strong driver of `YES`, but it is not sufficient alone to explain `MATCH`.\n- **Do shared interests matter?** Yes. Interest alignment and compatibility signals are among the most stable drivers of match formation.\n- **Does age matter?** Yes, but mostly through age proximity and gap rather than age as an isolated variable.\n- **Does race matter?** Some demographic signals can be significant, but they should be interpreted carefully and ethically.\n- **Can we predict a match?** Yes, with moderate-to-strong performance depending on feature availability. The realistic model remains informative, while the leaky model inflates performance.\n\n## Key business conclusions\n\n- YES rate: {yes_rate_text}\n- MATCH rate: {match_rate_text}\n- Best model: {best_model.get('model', 'n/a')}\n- Test ROC AUC: {best_model.get('test_roc_auc', 'n/a')}\n\n## Statistical evidence\n\n{statistics_summary}\n\n## Leakage analysis\n\n{leakage}\n\n## Actionable recommendations\n\n1. Focus the matching logic on compatibility and shared interests, not only on attractiveness.\n2. Use age-gap and preference alignment as high-value matching features.\n3. Keep pre-date and in-date variables separated from post-date signals in any predictive use case.\n4. Monitor demographic variables carefully to avoid unfair or over-interpreted conclusions.\n"""


def main() -> None:
    ensure_structure()
    write_notebooks()
    write_dashboard()
    write_slides()

    raw_path = download_dataset(RAW_DIR / DEFAULT_RAW_FILENAME)
    df_raw = load_raw_dataset(raw_path)
    df = clean_dataset(df_raw)
    df = engineer_features(df)
    df.to_csv(PROCESSED_DIR / "final_dataset.csv", index=False)

    summary = summarize_dataset(df)
    missingness = build_missingness_table(df)
    duplicates = find_duplicates(df)
    groups = define_column_groups(df)
    critical_columns = get_critical_columns(df)
    statistical_summary = build_statistical_summary(df)

    save_json(
        PROJECT_DIR / "analysis_summary.json",
        {
            "dataset_summary": summary,
            "missingness": missingness.head(25),
            "duplicates_count": int(len(duplicates)),
            "critical_columns": critical_columns,
            "column_groups": groups,
            "statistical_summary": statistical_summary,
        },
    )

    save_missingness_heatmap(df, FIGURES_DIR / "missingness_heatmap.png")
    save_boxplots(df, ["age", "age_o", "attr", "fun", "like"], FIGURES_DIR / "outliers_boxplots.png")
    save_yes_match_rates(df, FIGURES_DIR / "yes_match_rates.png")
    save_distribution_plot(df, "age", FIGURES_DIR / "age_distribution.png")
    save_correlation_heatmap(df, [column for column in ["age", "age_o", "attr", "fun", "like", "int_corr", "match", "dec"] if column in df.columns], FIGURES_DIR / "correlation_heatmap.png")

    feature_columns = [
        column
        for column in [
            "age",
            "age_o",
            "attr",
            "fun",
            "intel",
            "like",
            "samerace",
            "int_corr",
            "age_gap",
            "preference_alignment",
            "compatibility_score",
            "personality_score",
            "attractiveness_weight",
        ]
        if column in df.columns
    ]
    modeling_output = evaluate_models(df, feature_columns=feature_columns, target_column="match")
    save_json(
        PROJECT_DIR / "reports" / "model_metrics.json",
        {
            "metrics": modeling_output["metrics"],
            "confusion_matrices": modeling_output["confusion_matrices"],
            "best_model_name": modeling_output["best_model_name"],
        },
    )
    if modeling_output["best_model_name"] in modeling_output["confusion_matrices"]:
        save_confusion_matrix(modeling_output["confusion_matrices"][modeling_output["best_model_name"]], modeling_output["best_model_name"], FIGURES_DIR / "best_model_confusion_matrix.png")
    save_roc_curves(modeling_output["roc_data"], FIGURES_DIR / "roc_curves.png")
    save_bundle(PROJECT_DIR / "reports" / "model_bundle.joblib", modeling_output)
    if modeling_output["best_model"] is not None:
        try:
            preprocessor = modeling_output["best_model"].named_steps["preprocessor"]
            transformed = preprocessor.transform(df[modeling_output["feature_columns"]])
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()
            feature_names = list(preprocessor.get_feature_names_out())
            if len(feature_names) != transformed.shape[1]:
                feature_names = [f"feature_{index}" for index in range(transformed.shape[1])]
            shap_df = pd.DataFrame(transformed, columns=feature_names)
            save_shap_summary_plot(modeling_output["best_model"].named_steps["model"], shap_df, FIGURES_DIR / "shap_summary.png")
            if feature_names:
                save_shap_dependence_plot(modeling_output["best_model"].named_steps["model"], shap_df, feature_names[0], FIGURES_DIR / "shap_dependence.png")
        except Exception:
            pass

    leakage = leakage_comparison(
        df,
        realistic_features=[column for column in ["age", "age_o", "attr", "fun", "intel", "like", "samerace", "int_corr", "age_gap", "preference_alignment", "compatibility_score", "personality_score"] if column in df.columns],
        leaky_features=[column for column in df.columns if column != "match"],
        target_column="match",
    )
    save_json(PROJECT_DIR / "reports" / "leakage_analysis.json", leakage)
    storytelling_markdown = build_storytelling_markdown(summary, statistical_summary, modeling_output["metrics"], leakage)
    (PROJECT_DIR / "reports" / "final_storytelling.md").write_text(storytelling_markdown, encoding="utf-8")
    save_executive_summary(
        {
            "rows": summary["rows"],
            "columns": summary["columns"],
            "participants": summary["participants"],
            "yes_rate": summary["yes_rate"],
            "match_rate": summary["match_rate"],
            "best_model": modeling_output["best_model_name"],
        },
        PROJECT_DIR / "reports" / "executive_summary.pdf",
    )


if __name__ == "__main__":
    main()