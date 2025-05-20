import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier()
model.fit(X, y)

with open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)

pip install fastapi

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

class IrisInput(BaseModel):
    features: list  # Example: [5.1, 3.5, 1.4, 0.2]

@app.post("/predict")
def predict(data: IrisInput):
    prediction = model.predict([data.features])
    return {"prediction": int(prediction[0])}

pip install uvicorn

!uvicorn app:app --reload
