import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error,r2_score
temp=pd.read_csv(r"D:\Projects\Smart Crop Yield Predictor\Dataset\temp.csv")
print(temp)
temp.info()
print(temp.describe())
print(temp.isnull().sum())
print(temp.duplicated().sum())
print("Mean: ",temp['avg_temp'].mean())
print("Median: ",temp['avg_temp'].median())
temp['avg_temp']=temp['avg_temp'].fillna(temp['avg_temp'].mean())
plt.boxplot(temp['avg_temp'])
#plt.show()
temp.columns=temp.columns.str.strip()
print(temp.columns)
temp['country']=temp['country'].str.strip()
print(temp['country'].nunique())
#print(temp['year'].min(),temp['year'].max())#1743 2013
print(temp.columns)
print(temp.groupby(['country','year']).size().value_counts().head())
temp=temp.groupby(['year','country'],as_index=False)['avg_temp'].mean()
print(temp.groupby(['country','year']).size().value_counts().head())
print(temp.duplicated().sum())
ohe=OneHotEncoder(sparse_output=False,handle_unknown='ignore')
country_Enc=ohe.fit_transform(temp[['country']])
country_df=pd.DataFrame(
    country_Enc,
    columns=ohe.get_feature_names_out(['country'])
)
temp=pd.concat([temp,country_df],axis=1)
temp.drop('country',axis=1,inplace=True)
x=temp.drop(['avg_temp'],axis=1)
y=temp['avg_temp']
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
"""
rf=RandomForestRegressor(n_estimators=50,random_state=42, max_depth=15, n_jobs=-1)
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
mae=mean_absolute_error(y_test,y_pred)
print("Mean Absolute Error: ",mae)
rmse=root_mean_squared_error(y_test,y_pred)
print("Root Mean Squared Error:",rmse)
r2=r2_score(y_test,y_pred)
print("R2 Score: ",r2)
"""