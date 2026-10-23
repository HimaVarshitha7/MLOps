import time
import joblib
import pandas as pd
from train_model import FEATURES

def streaming_predict(records=10, delay_seconds=1):
    model = joblib.load("model.pkl")
    heldout = pd.read_csv("heldout_passengers.csv")
    if len(heldout) < records:
        raise ValueError(f"Need {records} held-out rows, found {len(heldout)}.")
    outputs = []
    for i in range(records):
        row = heldout[FEATURES].iloc[[i]]
        prediction = int(model.predict(row)[0])
        outputs.append(prediction)
        print(f"Incoming record {i+1}/{records}: predicted Survived={prediction}")
        if i < records - 1 and delay_seconds:
            time.sleep(delay_seconds)
    return outputs

if __name__ == "__main__":
    streaming_predict()
