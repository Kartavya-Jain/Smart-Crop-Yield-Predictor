from fastapi import FastAPI
import joblib
from basemodelcls import CropInput
app=FastAPI()
model=joblib.load("crop_yield_predictor_model.pkl")
encoders=joblib.load("encoders.pkl")
print(encoders.keys())
@app.get("/")
def Home():
    return{
        "message":"Testing API"
    }
@app.post("/predict")
def predict(data: CropInput):
    area_enc=encoders["Area"].transform([data.country][0])
    item_enc=encoders["Item"].transform([data.crop][0])
    unit_enc=encoders["Unit"].transform(["tonnes"])[0]
    features=[[
        area_enc,
        item_enc,
        data.year,
        data.pesticides,
        data.rainfall,
        data.temperature
    ]]
    prediction=model.predict(features)
    return{
        "predicted_yield":float(prediction[0])
    }