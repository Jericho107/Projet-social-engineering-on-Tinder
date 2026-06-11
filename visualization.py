from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


def save_missingness_heatmap(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    missing = df.isna().astype(int)
    plt.figure(figsize=(14, 8))
    sns.heatmap(missing.sample(min(len(missing), 300), random_state=42), cbar=False, cmap="mako")
    plt.title("Missing Values Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def save_boxplots(df: pd.DataFrame, columns: list[str], output_path: Path) -> Path:
    available = [column for column in columns if column in df.columns]
    if not available:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(available), 1, figsize=(12, max(3, 3 * len(available))))
    if len(available) == 1:
        axes = [axes]
    for axis, column in zip(axes, available):
        sns.boxplot(x=df[column], ax=axis, color="#4C78A8")
        axis.set_title(f"Boxplot: {column}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def save_yes_match_rates(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = {}
    for column in ["dec", "match"]:
        if column in df.columns:
            values[column] = df[column].mean()
    series = pd.Series(values).sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=series.index, y=series.values, palette="viridis")
    plt.ylabel("Rate")
    plt.title("YES and MATCH Rates")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def save_distribution_plot(df: pd.DataFrame, column: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    sns.histplot(df[column].dropna(), kde=True, bins=30, color="#F58518")
    plt.title(f"Distribution of {column}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def save_roc_curves(roc_data: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 7))
    for model_name, coordinates in roc_data.items():
        plt.plot(coordinates["fpr"], coordinates["tpr"], label=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def save_confusion_matrix(matrix: list[list[int]], model_name: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(np.array(matrix), annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def save_correlation_heatmap(df: pd.DataFrame, columns: list[str], output_path: Path) -> Path:
    available = [column for column in columns if column in df.columns]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[available].corr(numeric_only=True), cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def save_shap_summary_plot(model, data_frame: pd.DataFrame, output_path: Path, sample_size: int = 300) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import shap
    except Exception:
        return output_path

    sampled = data_frame.sample(min(len(data_frame), sample_size), random_state=42).copy()
    explainer = shap.Explainer(model.predict, sampled)
    shap_values = explainer(sampled)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, sampled, show=False, max_display=min(20, sampled.shape[1]))
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def save_shap_dependence_plot(model, data_frame: pd.DataFrame, feature_name: str, output_path: Path, sample_size: int = 300) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import shap
    except Exception:
        return output_path

    if feature_name not in data_frame.columns:
        return output_path

    sampled = data_frame.sample(min(len(data_frame), sample_size), random_state=42).copy()
    explainer = shap.Explainer(model.predict, sampled)
    shap_values = explainer(sampled)
    plt.figure(figsize=(10, 7))
    shap.dependence_plot(feature_name, shap_values.values, sampled, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path
