from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DATA_URL = "https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/Speed%20Dating%20Data.csv"
DEFAULT_RAW_FILENAME = "Speed Dating Data.csv"

AFTER_DATE_COLUMNS = {
    "dec",
    "match",
    "match_es",
    "satis_2",
    "length",
    "numdat_2",
    "you_call",
    "them_cal",
    "date_3",
    "numdat_3",
    "num_in_3",
}

RATING_COLUMNS = ["attr", "sinc", "intel", "fun", "amb", "shar"]
DECLARED_PREFERENCE_COLUMNS = ["attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1"]


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_dataset(destination: Path, url: str = RAW_DATA_URL) -> Path:
    ensure_directory(destination.parent)
    if not destination.exists():
        urllib.request.urlretrieve(url, destination.as_posix())
    return destination


def load_raw_dataset(raw_path: Path | str | None = None) -> pd.DataFrame:
    if raw_path is None:
        raise ValueError("A raw path is required.")
    path = Path(raw_path)
    if path.is_dir():
        path = path / DEFAULT_RAW_FILENAME
    if not path.exists():
        download_dataset(path)
    return pd.read_csv(path, encoding="ISO-8859-1")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = [re.sub(r"\s+", "_", str(col).strip().lower()) for col in renamed.columns]
    return renamed


def _looks_numeric(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False
    numeric_like = sample.str.match(r"^[\s\-+]?\d+([.,]\d+)?%?$").mean()
    return numeric_like >= 0.7


def coerce_numeric_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    converted = df.copy()
    for column in converted.columns:
        if converted[column].dtype == object and _looks_numeric(converted[column]):
            values = (
                converted[column]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .replace({"nan": np.nan, "None": np.nan, "": np.nan})
            )
            converted[column] = pd.to_numeric(values, errors="coerce")
    return converted


def build_missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
        }
    ).sort_values("missing_pct", ascending=False)


def find_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    duplicate_mask = df.duplicated(keep=False)
    if not duplicate_mask.any():
        return df.iloc[0:0].copy()
    return df.loc[duplicate_mask].copy()


def summarize_dataset(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "participants": int(df["iid"].nunique()) if "iid" in df.columns else None,
        "waves": int(df["wave"].nunique()) if "wave" in df.columns else None,
        "yes_rate": float(df["dec"].mean()) if "dec" in df.columns else None,
        "match_rate": float(df["match"].mean()) if "match" in df.columns else None,
    }


def define_column_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    before_date = []
    during_date = []
    after_date = []

    for column in df.columns:
        if column in AFTER_DATE_COLUMNS:
            after_date.append(column)
        elif (
            column in RATING_COLUMNS
            or column.startswith("attr")
            or column.startswith("sinc")
            or column.startswith("intel")
            or column.startswith("fun")
            or column.startswith("amb")
            or column.startswith("shar")
            or column.startswith("like")
            or column.startswith("prob")
            or column.startswith("met")
        ):
            during_date.append(column)
        else:
            before_date.append(column)

    return {
        "before_date": before_date,
        "during_date": during_date,
        "after_date": after_date,
    }


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = standardize_columns(df)
    cleaned = coerce_numeric_like_columns(cleaned)
    for column in cleaned.columns:
        if cleaned[column].dtype == object:
            cleaned[column] = cleaned[column].replace({"": np.nan, " ": np.nan, "nan": np.nan, "NaN": np.nan})
    return cleaned


def get_critical_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "dec",
        "match",
        "age",
        "age_o",
        "gender",
        "race",
        "race_o",
        "samerace",
        "attr",
        "sinc",
        "intel",
        "fun",
        "amb",
        "shar",
        "like",
        "prob",
        "met",
        "attr1_1",
        "sinc1_1",
        "intel1_1",
        "fun1_1",
        "amb1_1",
        "shar1_1",
    ]
    return [column for column in candidates if column in df.columns]


def export_json_serializable(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Series):
        return value.astype(object).where(pd.notna(value), None).tolist()
    if isinstance(value, pd.DataFrame):
        return value.astype(object).where(pd.notna(value), None).to_dict(orient="records")
    if isinstance(value, (list, tuple)):
        return [export_json_serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): export_json_serializable(val) for key, val in value.items()}
    return value


def save_json(path: Path, payload: dict) -> Path:
    ensure_directory(path.parent)
    path.write_text(json.dumps(export_json_serializable(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
