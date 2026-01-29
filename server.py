from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("stress_level_model_human_like.pkl")

class SensorInput(BaseModel):
    ADXL345_X: float
    ADXL345_Y: float
    ADXL345_Z: float
    HeartRate: float
    SpO2: float
    LightLux: float
    GSR: float

@app.post("/predict")
def predict(data: SensorInput):
    df = pd.DataFrame([{
        "ADXL345_X": data.ADXL345_X,
        "ADXL345_Y": data.ADXL345_Y,
        "ADXL345_Z": data.ADXL345_Z,
        "MAX30102_HeartRate": data.HeartRate,
        "MAX30102_SpO2": data.SpO2,
        "BH1750_Light(lux)": data.LightLux,
        "GSR_Value": data.GSR
    }])

    pred = model.predict(df)[0]

    return {"stress_level": pred}
