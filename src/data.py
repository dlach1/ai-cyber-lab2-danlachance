from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import basic_clean_text, find_label_column, find_text_column, normalize_labels


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def prepare_data(
    path: str | Path,
    label_col: Optional[str] = None,
    text_col: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    df = load_dataset(path)

    label_col = find_label_column(df, label_col)
    text_col = find_text_column(df, text_col)

    df = df[[text_col, label_col]].dropna()
    df = df.drop_duplicates()

    df[text_col] = basic_clean_text(df[text_col])
    df[label_col] = normalize_labels(df[label_col])

    X = df[text_col]
    y = df[label_col]

    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return X_train, X_test, y_train, y_test
