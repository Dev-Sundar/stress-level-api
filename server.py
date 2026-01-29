from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
import numpy as np

app = FastAPI()

# Load Model
model = joblib.load("stress_level_model_human_like.pkl")

# Data Schema
class SensorData(BaseModel):
    ADXL345_X: float
    ADXL345_Y: float
    ADXL345_Z: float
    HeartRate: float
    SpO2: float
    LightLux: float
    GSR: float

@app.post("/predict")
def predict(data: SensorData):

    sample = np.array([[
        data.ADXL345_X,
        data.ADXL345_Y,
        data.ADXL345_Z,
        data.HeartRate,
        data.SpO2,
        data.LightLux,
        data.GSR
    ]])

    pred = model.predict(sample)[0]

    return { "stress_level": str(pred) }

# Run server: uvicorn server:app --reload
