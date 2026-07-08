from fastapi import FastAPI
import joblib
from basemodelcls import CropInput
import pandas as pd
base=pd.read_csv(r"Smart_Crop_Yield_Predictor.csv")
app=FastAPI()
model=joblib.load("crop_yield_predictor_model.pkl")
encoders=joblib.load("encoders.pkl")
print(encoders.keys())
print(encoders["Area"].classes_)
print(model.n_features_in_)
@app.get("/")
def Home():
    return{
        "message":"Testing API"
    }
@app.post("/predict")
def predict(data: CropInput):
    country=data.country.lower()
    crop=data.crop.lower()
    area_enc=encoders["Area"].transform([country])[0]
    item_enc=encoders["Item"].transform([crop])[0]
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
@app.get("/crops/{country}")
def get_crops(country:str):
    country=country.lower()
    if country not in encoders['Area'].classes_:
        return {"error":"Country not found"}
    country_enc=encoders['Area'].transform([country])[0]
    crops=base[base['Area']==country_enc]['Item'].unique()
    crop_names=encoders['Item'].inverse_transform(crops)
    return {"crops": list(crop_names)}
@app.get("/countries")
def get_countries():
    return{"countries": list(encoders['Area'].classes_)}