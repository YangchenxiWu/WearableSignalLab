"""Train and evaluate a baseline human activity classifier."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"


def main() -> None:
    features_path = OUTPUT_DIR / "features.csv"
    if not features_path.exists():
        raise FileNotFoundError(
            "Feature file not found. Run: python3 scripts/02_filter_and_features.py"
        )

    df = pd.read_csv(features_path)
    metadata_cols = {"split", "window_id", "activity_id", "activity"}
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()

    x_train = train[feature_cols]
    y_train = train["activity"]
    x_test = test[feature_cols]
    y_test = test["activity"]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred)
    labels = sorted(df["activity"].unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    metrics = {
        "classifier": "RandomForestClassifier",
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": int(len(feature_cols)),
        "labels": labels,
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Wrote metrics: {OUTPUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
