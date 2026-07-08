import joblib
import pandas as pd
model=joblib.load("crop_yield_predictor_model.pkl")
encoders=joblib.load("encoders.pkl")
print(type(model))
print(model.n_estimators)
print(model.max_depth)