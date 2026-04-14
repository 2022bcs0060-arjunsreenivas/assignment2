from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import numpy as np
import mlflow.pyfunc
import os

app=FastAPI()
model = joblib.load("model.pkl")

class Request(BaseModel):
    freq_7d: int
    freq_30d: int
    freq_90d: int
    complaint_count: int
    avg_gap: float
    charge_diff: float

@app.get("/")
def home():
    return {"message":"API started successfully"}

@app.post("/predict")
def predict(request: Request):
    req = np.array([[
        request.freq_7d,
        request.freq_30d,
        request.freq_90d,
        request.complaint_count,
        request.avg_gap,
        request.charge_diff,
    ]])

    prediction = model.predict(req)
    print(prediction)

    output=["HIGH","LOW","MEDIUM"]

    return {
        "Name":"ARJUN SREENIVAS",
        "Roll No":"2022BCS0060",
        "predicted value":output[int(prediction[0])],
    }