import joblib
import pandas as pd
from train_model import FEATURES

def batch_predict():
    model = joblib.load("model.pkl")
    heldout = pd.read_csv("heldout_passengers.csv")
    if len(heldout) < 20:
        raise ValueError(f"Need 20 held-out rows, found {len(heldout)}.")
    batch = heldout[FEATURES].iloc[:20].copy()
    results = batch.copy()
    results["Predicted_Survived"] = model.predict(batch).astype(int)
    results["Actual_Survived"] = heldout["Survived"].iloc[:20].astype(int).values
    results.to_csv("batch_predictions.csv", index=False)
    print(results.to_string(index=False))
    print("Saved batch_predictions.csv")
    return results

if __name__ == "__main__":
    batch_predict()
