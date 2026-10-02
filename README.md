<div align="center">

# Speed Dating Behavioral Analytics

### Statistical inference · leakage-aware machine learning · explainability · decision boundaries

**Python · Pandas · SciPy · Statsmodels · scikit-learn · XGBoost · SHAP · Streamlit · CI**

**Pretoria BI — Data · Intelligence · Performance**

</div>

---

## Analytical question

> **Which observed signals are associated with a participant saying YES, and which signals remain useful when the harder outcome is a mutual MATCH?**

This repository analyses the public Speed Dating experiment dataset as a behavioral-analytics case study. It separates descriptive patterns, statistical evidence and predictive performance instead of treating correlation, feature importance and causality as interchangeable.

No production dating system, psychological diagnosis or causal claim is implied.

---

## Why this project exists

The value of the case study is not the theme. It demonstrates several analytical problems that recur in commercial work:

- high missingness and survey-style data quality;
- stated preference versus observed behaviour;
- categorical and continuous statistical testing;
- effect sizes in addition to p-values;
- classification with class imbalance;
- feature leakage;
- explainability;
- translating model output into bounded decision language.

---

## Evidence model

```text
RAW EXPERIMENT DATA
        ↓
DATA AUDIT
        ↓
BEHAVIOURAL EDA
        ↓
STATISTICAL TESTS + EFFECT SIZES
        ↓
FEATURE CONTRACT
        ↓
BASELINE / CANDIDATE MODELS
        ↓
LEAKAGE AUDIT
        ↓
OUT-OF-SAMPLE METRICS
        ↓
EXPLAINABILITY
        ↓
BOUNDED INTERPRETATION
```

### Fail-closed feature governance

`src/governance.py` explicitly rejects:

- the prediction target itself;
- known post-outcome / partner-outcome fields;
- invalid target states;
- impossible probability outputs.

CI deliberately injects target leakage and requires the audit to fail.

---

## Business-style questions

1. Do declared preferences align with observed decisions?
2. Which variables have the strongest association with a YES?
3. Which signals survive when the target becomes mutual MATCH?
4. How much predictive performance disappears when post-outcome information is removed?
5. Are statistically significant effects also practically meaningful?
6. Which model explanations are stable enough to discuss, and which are merely model-specific?

---

## Repository structure

```text
.
├── data/
├── notebooks/
├── reports/
├── src/
│   └── governance.py
├── tests/
├── app.py
├── build_project.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Validation

Static and governance checks:

```bash
python -m pip install -e ".[dev]"
python -m py_compile app.py src/governance.py
ruff check app.py src/governance.py tests
pytest -q
```

Run the Streamlit interface after generated artifacts are available:

```bash
streamlit run app.py
```

The root Streamlit application now resolves repository assets from the repository root rather than its parent directory.

---

## Interpretation boundary

This project may support statements such as:

- a variable is associated with an outcome in this dataset;
- a model improves or degrades under a documented feature set;
- an effect has a measured magnitude;
- removing leaky information changes out-of-sample performance.

It does **not** support claims that:

- a feature causes attraction or matching;
- observed demographic relationships generalise to all populations;
- a high-performing leaky model is deployable;
- SHAP values establish causal influence.

---

## Proof standard

> **A result is only portfolio evidence when its assumptions, evaluation path and failure mode are inspectable.**

The repository is being consolidated under the same Pretoria BI standard used by the flagship projects: **Understand · Decide · Act · Measure**.
