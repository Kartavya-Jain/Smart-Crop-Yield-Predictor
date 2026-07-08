import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error

yield_df=pd.read_csv(r"D:\Projects\Smart Crop Yield Predictor\Dataset\yield_df.csv")
print(yield_df)
print(yield_df.info())
print(yield_df.describe())
print(yield_df.isnull().sum())
print(yield_df.duplicated().sum())
#plt.boxplot(df['average_rain_fall_mm_per_year'])
#plt.show()
print(yield_df)
print(yield_df['Year'].nunique())
print(yield_df['hg/ha_yield'].nunique())
print(yield_df['average_rain_fall_mm_per_year'].nunique())
print(yield_df['pesticides_tonnes'].nunique())
print(yield_df['avg_temp'].nunique())
print(yield_df[['Area','Item','Year']].duplicated().sum())
print(yield_df)
ohe_area=OneHotEncoder(sparse_output=False, handle_unknown='ignore')
Area_Enc=ohe_area.fit_transform(yield_df[['Area']])
area_df=pd.DataFrame(
    Area_Enc,
    columns=ohe_area.get_feature_names_out(['Area']),
    index=yield_df.index
)
ohe_item=OneHotEncoder(sparse_output=False, handle_unknown='ignore')
Item_Enc=ohe_item.fit_transform(yield_df[['Item']])
item_df=pd.DataFrame(
    Item_Enc,
    columns=ohe_item.get_feature_names_out(['Item']),
    index=yield_df.index
)
df=pd.concat([yield_df,area_df,item_df],axis=1)
#print(yield_df['Year'].min(),yield_df['Year'].max())
print(df.groupby(['Area','Item','Year','avg_temp']).size().value_counts().head())
df.drop(['Area','Item'],axis=1,inplace=True)
print(df.columns)
print(df.shape)
x=df.drop(['hg/ha_yield'],axis=1)
y=df['hg/ha_yield']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#lr=LinearRegression()
#lr.fit(x_train,y_train)
#y_pred=lr.predict(x_test)
#r2=r2_score(y_test,y_pred)
#mse=mean_squared_error(y_test,y_pred)
#rmse=root_mean_squared_error(y_test,y_pred)
#print("R2 Score: ",r2)
#print("Mean Squared Error: ",mse)
#print("Root Mean Squared Error: ",rmse)
model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
r2=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=root_mean_squared_error(y_test,y_pred)
print("R2 Score: ",r2)
print("Mean Squared Error: ",mse)
print("Root Mean Squared Error: ",rmse)
print("Train R2:", model.score(x_train, y_train))
print("Test R2:", model.score(x_test, y_test))