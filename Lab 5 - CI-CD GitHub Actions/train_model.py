
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

FEATURES = [
    "CurriculumCoverage", "IndustryDemand", "JobFrequency",
    "ExpertImportance", "Programming", "DataStructures",
    "Algorithms", "Database", "WebDevelopment", "Cloud",
    "AI_ML", "Communication", "ProblemSolving", "Teamwork",
    "Projects", "Internship", "Certifications",
    "CurriculumAlignment", "IndustryRequirement", "SkillGap"
]
TARGET = "GapStatus"


def train_model(data_path, output_dir="artifacts"):
    df = pd.read_csv(data_path)

    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=[TARGET])
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET]

    if y.nunique() < 2:
        raise ValueError("GapStatus must contain at least two classes.")

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), FEATURES)
    ])

    model = Pipeline([
        ("preprocessor", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_weighted": float(precision_score(
            y_test, predictions, average="weighted", zero_division=0
        )),
        "recall_weighted": float(recall_score(
            y_test, predictions, average="weighted", zero_division=0
        )),
        "f1_weighted": float(f1_score(
            y_test, predictions, average="weighted", zero_division=0
        ))
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "skill_gap_model.joblib")

    with open(output / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation metrics:")
    print(json.dumps(metrics, indent=4))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("\nSample predictions:")
    print(model.predict(X_test.head(5)))
    print("\nModel saved successfully.")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="cse_skill_gap.csv")
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()

    train_model(args.data, args.output)
