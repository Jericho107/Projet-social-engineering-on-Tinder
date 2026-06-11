from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def cohens_d(sample_a: pd.Series, sample_b: pd.Series) -> float:
    a = pd.to_numeric(sample_a, errors="coerce").dropna()
    b = pd.to_numeric(sample_b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_std = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    if pooled_std == 0 or np.isnan(pooled_std):
        return float("nan")
    return (a.mean() - b.mean()) / pooled_std


def cramers_v(contingency: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(contingency)[0]
    n = contingency.to_numpy().sum()
    if n == 0:
        return float("nan")
    r, k = contingency.shape
    return np.sqrt((chi2 / n) / max(min(k - 1, r - 1), 1))


def t_test_by_decision(df: pd.DataFrame, columns, decision_column: str = "dec") -> pd.DataFrame:
    rows = []
    for column in columns:
        if column not in df.columns or decision_column not in df.columns:
            continue
        yes = pd.to_numeric(df.loc[df[decision_column] == 1, column], errors="coerce").dropna()
        no = pd.to_numeric(df.loc[df[decision_column] == 0, column], errors="coerce").dropna()
        if len(yes) < 2 or len(no) < 2:
            continue
        statistic, pvalue = stats.ttest_ind(yes, no, equal_var=False, nan_policy="omit")
        rows.append(
            {
                "variable": column,
                "yes_mean": yes.mean(),
                "no_mean": no.mean(),
                "t_stat": statistic,
                "p_value": pvalue,
                "cohens_d": cohens_d(yes, no),
            }
        )
    return pd.DataFrame(rows).sort_values("p_value", ascending=True)


def anova_by_factor(df: pd.DataFrame, numeric_column: str, factor_column: str) -> dict:
    if numeric_column not in df.columns or factor_column not in df.columns:
        return {}
    working = df[[numeric_column, factor_column]].copy()
    working[numeric_column] = pd.to_numeric(working[numeric_column], errors="coerce")
    groups = [group[numeric_column].dropna().values for _, group in working.groupby(factor_column)]
    groups = [group for group in groups if len(group) >= 2]
    if len(groups) < 2:
        return {}
    statistic, pvalue = stats.f_oneway(*groups)
    overall_mean = working[numeric_column].mean()
    ss_between = sum(len(group) * (group.mean() - overall_mean) ** 2 for group in groups)
    ss_total = ((working[numeric_column] - overall_mean) ** 2).sum()
    eta_squared = ss_between / ss_total if ss_total not in (0, np.nan) and pd.notna(ss_total) else np.nan
    return {
        "numeric_column": numeric_column,
        "factor_column": factor_column,
        "f_stat": float(statistic),
        "p_value": float(pvalue),
        "eta_squared": float(eta_squared) if pd.notna(eta_squared) else np.nan,
    }


def chi_square_test(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    if col_a not in df.columns or col_b not in df.columns:
        return {}
    contingency = pd.crosstab(df[col_a], df[col_b])
    if contingency.empty:
        return {}
    chi2, pvalue, dof, expected = stats.chi2_contingency(contingency)
    return {
        "col_a": col_a,
        "col_b": col_b,
        "chi2": float(chi2),
        "p_value": float(pvalue),
        "dof": int(dof),
        "cramers_v": float(cramers_v(contingency)),
    }


def build_statistical_summary(df: pd.DataFrame) -> dict:
    t_test_columns = [column for column in ["attr", "fun", "like", "shar", "intel", "sinc"] if column in df.columns]
    ttests = t_test_by_decision(df, t_test_columns).to_dict(orient="records") if t_test_columns else []
    anovas = []
    for numeric_column in [column for column in ["age", "age_o"] if column in df.columns]:
        for factor_column in [column for column in ["race", "field", "gender"] if column in df.columns]:
            result = anova_by_factor(df, numeric_column, factor_column)
            if result:
                anovas.append(result)
    chi_squares = []
    for left, right in [("gender", "match"), ("race", "match"), ("samerace", "match")]:
        result = chi_square_test(df, left, right)
        if result:
            chi_squares.append(result)
    return {
        "ttests": ttests,
        "anovas": anovas,
        "chi_squares": chi_squares,
    }
