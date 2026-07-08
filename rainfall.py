import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,root_mean_squared_error,r2_score
rainfall=pd.read_csv(r"D:\Projects\Smart Crop Yield Predictor\Dataset\rainfall.csv")
print(rainfall)
rainfall.columns=rainfall.columns.str.strip()
rainfall['Area']=rainfall['Area'].str.strip()
rainfall.info()
print(rainfall.describe())
print(rainfall.isnull().sum())
print(rainfall.duplicated().sum())
rainfall['average_rain_fall_mm_per_year']=rainfall['average_rain_fall_mm_per_year'].str.extract('(\d+\.?\d*)').astype(float)
rainfall['average_rain_fall_mm_per_year']=rainfall['average_rain_fall_mm_per_year'].fillna(rainfall['average_rain_fall_mm_per_year'].median())
print(rainfall['average_rain_fall_mm_per_year'].head(10))
plt.boxplot(rainfall['average_rain_fall_mm_per_year'])
#plt.show()
Q1=rainfall['average_rain_fall_mm_per_year'].quantile(0.25)
Q3=rainfall['average_rain_fall_mm_per_year'].quantile(0.75)
IQR=Q3-Q1
lower=Q1-1.5*IQR
upper=Q3+1.5*IQR
filter=(rainfall['average_rain_fall_mm_per_year']<=upper) & (rainfall['average_rain_fall_mm_per_year']>=lower)
rainfall=rainfall[filter]
plt.boxplot(rainfall['average_rain_fall_mm_per_year'])
#plt.show()
print(rainfall.columns)
print(rainfall['Year'].min(),rainfall['Year'].max())#1985 2017
le=LabelEncoder()
rainfall['Area']=le.fit_transform(rainfall['Area'])
x=rainfall[['Area','Year']]
y=rainfall['average_rain_fall_mm_per_year']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
lr=LinearRegression()
lr.fit(x_train,y_train)
lr_pred=lr.predict(x_test)
mae=mean_absolute_error(y_test,lr_pred)
print("Mean Absolute Error: ",mae)
rmse=root_mean_squared_error(y_test,lr_pred)
print("Root Mean Squared Error: ", rmse)
r2=r2_score(y_test,lr_pred)
print("R2 score: ",r2)
rf=RandomForestRegressor(n_estimators=50,random_state=42, max_depth=15, n_jobs=-1)
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
mae=mean_absolute_error(y_test,y_pred)
print("Mean Absolute Error: ",mae)
rmse=root_mean_squared_error(y_test,y_pred)
print("Root Mean Squared Error:",rmse)
r2=r2_score(y_test,y_pred)
print("R2 Score: ",r2)