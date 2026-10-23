# CLA 4 — ML Model Deployment and FastAPI

Dataset: Titanic — Machine Learning from Disaster, using the labelled Kaggle `train.csv`: https://www.kaggle.com/competitions/titanic/data

## Google Colab
1. Open `CLA4_Colab_Notebook.ipynb` in Google Colab.
2. Download `train.csv` from Kaggle and upload it when prompted.
3. Run cells from top to bottom. The notebook trains once, saves `model.pkl`, creates `heldout_passengers.csv` and `batch_predictions.csv`, demonstrates online/batch/streaming inference, and tests FastAPI endpoints.

## Run as Python scripts
Install packages with `pip install -r requirements.txt`, then:
```
python train_model.py --data train.csv --model model.pkl
python online_inference.py
python batch_inference.py
python streaming_inference.py
uvicorn app:app --reload
```
Open `http://127.0.0.1:8000/docs` for Swagger UI. In another terminal, run `python client.py`.

The same saved pipeline is reused for all inference types. Accuracy is printed at training time and depends on the dataset and split. Inference examples use the held-out test split, which is not used to train the model.
