import argparse
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

NUMERIC = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL = ["Sex", "Embarked"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Survived"

def train_model(data_path="train.csv", model_path="model.pkl"):
    df = pd.read_csv(data_path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.dropna(subset=[TARGET]).copy()
    X, y = df[FEATURES], df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    prep = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), NUMERIC),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), CATEGORICAL)
    ])
    model = Pipeline([
        ("preprocessing", prep),
        ("classifier", RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    joblib.dump(model, model_path)
    heldout = X_test.copy()
    heldout["Survived"] = y_test.values
    heldout.to_csv("heldout_passengers.csv", index=False)
    print(f"Test accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print(f"Saved model to {model_path}")
    print(classification_report(y_test, pred, zero_division=0))
    return model, accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="train.csv")
    parser.add_argument("--model", default="model.pkl")
    args = parser.parse_args()
    train_model(args.data, args.model)
