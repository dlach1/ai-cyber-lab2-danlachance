from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from .data import prepare_data
from .utils import plot_confusion_matrix, save_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate phishing detector baseline.")
    parser.add_argument("--data-path", required=True, help="Path to CSV/TSV dataset.")
    parser.add_argument("--label-col", default=None, help="Label column name.")
    parser.add_argument("--text-col", default=None, help="Text/URL column name.")
    parser.add_argument("--model-path", default="results/model.joblib", help="Model path.")
    parser.add_argument("--metrics-out", default="results/metrics.json")
    parser.add_argument("--cm-out", default="results/confusion_matrix.png")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, X_test, _, y_test = prepare_data(
        args.data_path,
        label_col=args.label_col,
        text_col=args.text_col,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model = joblib.load(args.model_path)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    save_metrics(metrics, args.metrics_out)

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, labels=["benign", "phishing"], path=args.cm_out)

    print(f"Metrics saved to {Path(args.metrics_out)}")
    print(f"Confusion matrix saved to {Path(args.cm_out)}")


if __name__ == "__main__":
    main()
