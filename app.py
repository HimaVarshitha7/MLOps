from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

MODEL_PATH = Path(__file__).with_name("model.pkl")
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
app = FastAPI(title="Titanic Survival Prediction API", version="1.0.0")

class Passenger(BaseModel):
    model_config = ConfigDict(extra="forbid")
    Pclass: int = Field(ge=1, le=3)
    Sex: str
    Age: float | None = Field(default=None, ge=0, le=120)
    SibSp: int = Field(ge=0, le=20)
    Parch: int = Field(ge=0, le=20)
    Fare: float = Field(ge=0)
    Embarked: str | None = "S"

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(passenger: Passenger):
    if model is None:
        raise HTTPException(status_code=503, detail="model.pkl not found; run train_model.py first")
    row = pd.DataFrame([passenger.model_dump()])[[
        "Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"
    ]]
    pred = int(model.predict(row)[0])
    return {"prediction": pred, "label": "Survived" if pred else "Did not survive"}
