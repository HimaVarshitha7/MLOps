
import pandas as pd
import pytest

from train_model import (
    FEATURES,
    TARGET,
    train_model
)



def make_sample_data():
    data = {
        feature: [
            20, 40, 60, 80, 90, 50, 30, 70, 85, 45,
            25, 55, 65, 75, 35, 95, 15, 48, 68, 88
        ]
        for feature in FEATURES
    }

    data[TARGET] = [
        "Low", "Low", "Medium", "High", "High",
        "Medium", "Low", "High", "High", "Medium",
        "Low", "Medium", "Medium", "High", "Low",
        "High", "Low", "Medium", "Medium", "High"
    ]

    return pd.DataFrame(data)



def test_missing_column(tmp_path):
    df = make_sample_data().drop(columns=["Programming"])
    file_path = tmp_path / "sample.csv"
    df.to_csv(file_path, index=False)

    with pytest.raises(ValueError, match="Missing columns"):
        train_model(file_path, tmp_path / "output")


def test_training_creates_model_and_metrics(tmp_path):
    df = make_sample_data()
    file_path = tmp_path / "sample.csv"
    df.to_csv(file_path, index=False)

    output_dir = tmp_path / "output"
    metrics = train_model(file_path, output_dir)

    assert "accuracy" in metrics
    assert "f1_weighted" in metrics
    assert (output_dir / "skill_gap_model.joblib").exists()
    assert (output_dir / "metrics.json").exists()
