from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import prepare_data
from .utils import ensure_dir


def build_model(ngram_min: int, ngram_max: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(ngram_min, ngram_max),
                    min_df=2,
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, solver="liblinear")),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train phishing detector baseline.")
    parser.add_argument("--data-path", required=True, help="Path to CSV/TSV dataset.")
    parser.add_argument("--label-col", default=None, help="Label column name.")
    parser.add_argument("--text-col", default=None, help="Text/URL column name.")
    parser.add_argument("--model-out", default="results/model.joblib", help="Model path.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--ngram-min", type=int, default=3)
    parser.add_argument("--ngram-max", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X_train, X_test, y_train, y_test = prepare_data(
        args.data_path,
        label_col=args.label_col,
        text_col=args.text_col,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model = build_model(args.ngram_min, args.ngram_max)
    model.fit(X_train, y_train)

    model_out = Path(args.model_out)
    ensure_dir(model_out.parent)
    joblib.dump(model, model_out)

    print(f"Trained model saved to {model_out}")
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")


if __name__ == "__main__":
    main()
