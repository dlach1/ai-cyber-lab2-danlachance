from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABEL_COL_CANDIDATES = [
    "label",
    "is_phishing",
    "phishing",
    "target",
    "class",
    "y",
]

TEXT_COL_CANDIDATES = [
    "url",
    "email",
    "text",
    "body",
    "content",
    "message",
    "subject",
]

POSITIVE_LABELS = {
    "phish",
    "phishing",
    "malicious",
    "spam",
    "fraud",
    "defacement",
    "1",
    "true",
    "yes",
}


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def find_label_column(df: pd.DataFrame, label_col: Optional[str] = None) -> str:
    if label_col:
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in dataset.")
        return label_col
    inferred = find_column(df, LABEL_COL_CANDIDATES)
    if not inferred:
        raise ValueError(
            "Could not infer label column. Pass --label-col with the correct column name."
        )
    return inferred


def find_text_column(df: pd.DataFrame, text_col: Optional[str] = None) -> str:
    if text_col:
        if text_col not in df.columns:
            raise ValueError(f"Text column '{text_col}' not found in dataset.")
        return text_col
    inferred = find_column(df, TEXT_COL_CANDIDATES)
    if not inferred:
        raise ValueError(
            "Could not infer text column. Pass --text-col with the correct column name."
        )
    return inferred


def basic_clean_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace(r"\s+", " ", regex=True)
    )


def normalize_labels(labels: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(labels):
        return labels.astype(int)
    return (
        labels.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda v: 1 if v in POSITIVE_LABELS else 0)
        .astype(int)
    )


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
