import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
yield_data=pd.read_csv(r"D:\Projects\Smart Crop Yield Predictor\Dataset\yield.csv")
print(yield_data)
print(yield_data.info())
print(yield_data.describe())
print(yield_data.isnull().sum())
print(yield_data.duplicated().sum())
yield_data.boxplot(column='Value', by='Item', rot=45)
#plt.show()
print(yield_data['Item'].nunique())
print(yield_data['Item'].unique())
yield_data['Area']=yield_data['Area'].str.strip()
yield_data.drop(['Domain Code','Domain','Element Code','Element','Unit'],axis=1,inplace=True)
#print(yield_data['Year'].min(),yield_data['Year'].max())#1961 2016
print(yield_data.groupby(['Area','Item','Year']).size().value_counts().head())
print(yield_data['Area'].nunique())
print(yield_data.groupby('Item')['Value'].describe())
print(yield_data.isnull().sum())
print(yield_data.duplicated().sum())
print(yield_data.shape)
ohe_area=OneHotEncoder(sparse_output=False, handle_unknown='ignore')
Area_Enc=ohe_area.fit_transform(yield_data[['Area']])
area_df=pd.DataFrame(
    Area_Enc,
    columns=ohe_area.get_feature_names_out(['Area'])
)
ohe_item=OneHotEncoder(sparse_output=False, handle_unknown='ignore')
Item_Enc=ohe_item.fit_transform(yield_data[['Item']])
item_df=pd.DataFrame(
    Item_Enc,
    columns=ohe_item.get_feature_names_out(['Item'])
)
yield_data=pd.concat([yield_data,area_df,item_df],axis=1)
yield_data.drop(['Area','Item','Area Code','Item Code'],axis=1,inplace=True)
print(yield_data)
x=yield_data.drop('Value',axis=1)
y=yield_data['Value']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
r2=r2_score(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=root_mean_squared_error(y_test,y_pred)
print("R2 Score: ",r2)
print("Mean Absolute Error: ",mae)
print("Root Mean Squared Error: ",rmse)
model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
r2=r2_score(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=root_mean_squared_error(y_test,y_pred)
print("R2 Score: ",r2)
print("Mean Absolute Error: ",mae)
print("Root Mean Squared Error: ",rmse)
print("Train R2:", model.score(x_train, y_train))
print("Test R2:", model.score(x_test, y_test))