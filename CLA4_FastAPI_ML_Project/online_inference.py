import joblib
import pandas as pd
from train_model import FEATURES

def online_predict():
    model = joblib.load("model.pkl")
    heldout = pd.read_csv("heldout_passengers.csv")
    row = heldout[FEATURES].iloc[[0]]
    prediction = int(model.predict(row)[0])
    print("Online inference — one unseen passenger")
    print(row.to_string(index=False))
    print("Prediction:", prediction, "-", "Survived" if prediction else "Did not survive")
    print("Actual held-out label:", int(heldout["Survived"].iloc[0]))
    return prediction

if __name__ == "__main__":
    online_predict()
