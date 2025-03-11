from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

EARNINGS = 0.1

model = joblib.load('model.joblib')

app = FastAPI()

class Instance(BaseModel):
    age: int
    sex: bool
    bmi: float
    children: int
    smoker: bool
    region_northeast: bool
    region_northwest: bool
    region_southeast: bool
    region_southwest: bool

@app.post("/predict/")
async def predict(instance: Instance):
    features = np.array([
        instance.age,
        instance.sex,
        instance.bmi,
        instance.children,
        instance.smoker,
        instance.region_northeast,
        instance.region_northwest,
        instance.region_southeast,
        instance.region_southwest
    ]).reshape(1, -1)

    prediction = model.predict(features)

    price = (1 + EARNINGS) * prediction[0]

    return {"price": round(price, 2)}
