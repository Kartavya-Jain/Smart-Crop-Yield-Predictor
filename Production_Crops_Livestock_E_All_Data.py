import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import joblib
production=pd.read_csv(r"D:\Projects\Smart Crop Yield Predictor\Dataset\Production_Crops_Livestock_E_All_Data.csv",low_memory=False)
pesticides=pd.read_csv(r"D:\Projects\Smart Crop Yield Predictor\Dataset\pesticides.csv")
rainfall=pd.read_csv(r"D:\Projects\Smart Crop Yield Predictor\Dataset\rainfall.csv")
temp=pd.read_csv(r"D:\Projects\Smart Crop Yield Predictor\Dataset\temp.csv")
yield_data=pd.read_csv(r"D:\Projects\Smart Crop Yield Predictor\Dataset\yield.csv")
yield_df=pd.read_csv(r"D:\Projects\Smart Crop Yield Predictor\Dataset\yield_df.csv")
production.columns=production.columns.str.strip()
drop_cols=[col for col in production.columns if col.endswith('F') or col.endswith('N')]
production=production.drop(columns=drop_cols,axis=1)
year_cols=[col for col in production.columns if col.startswith('Y')]
production_long=production.melt(
    id_vars=['Area', 'Item', 'Element', 'Unit'],
    value_vars=year_cols,
    var_name='Year',
    value_name='production_tonnes'
)
production_long['Year']=(production_long['Year'].str.replace('Y','',regex=True).astype(int))
for df in [pesticides,production,rainfall,temp,yield_data,yield_df]:
    df.columns=df.columns.str.strip()
    if 'Area' in df.columns:
        df['Area']=df['Area'].str.strip().str.lower()
    if 'Item' in df.columns:
        df['Item']=df['Item'].str.strip().str.lower()
    if 'Year' in df.columns:
        df['Year']=pd.to_numeric(df['Year'])

base=production_long.copy()
base=base[base['Element']=='Production']
for col in ['Area','Item']:
    base[col]=base[col].str.strip().str.lower()
#1. Merge pesticides
#print(pesticides.columns)
#print(pesticides.head())
base=base.merge(
    pesticides[['Area','Year','Value']],
    on=['Area','Year'],
    how='left'
)
#print("After pesticides:", base.shape)
base.rename(columns={'Value':'pesticides_tonnes'},inplace=True)
#2. Merge rainfall
rainfall.columns=rainfall.columns.str.strip()
#print(rainfall.columns.tolist())
base=base.merge(
    rainfall[['Area','Year','average_rain_fall_mm_per_year']],
    on=['Area','Year'],
    how='left'
)
base['average_rain_fall_mm_per_year']=pd.to_numeric(base['average_rain_fall_mm_per_year'],errors='coerce')
#print("After rainfall:", base.shape)
#3. Merge temp
temp=temp.rename(
    columns={
        'country':'Area',
        'year':'Year'
    }
)
temp['Area']=temp['Area'].str.strip().str.lower()
base=base.merge(
    temp[['Area','Year','avg_temp']],
    on=['Area','Year'],
    how='left'
)
#print("After temp:", base.shape)
#5. Merge production
#print(yield_df.columns.tolist())
production_long=production_long[production_long['Element']=='Production']
#print(production_long.duplicated(['Area','Item','Year']).sum())
base=base.merge(
    yield_df[['Area','Item','Year','hg/ha_yield']],
    on=['Area','Item','Year'],
    how='left'
)
#print("After yield:", base.shape)
#print(base['hg/ha_yield'].isnull().sum())
base['production_tonnes']=base['production_tonnes'].fillna(base['production_tonnes'].mean())
#print(base.isnull().sum())
base=base.drop(columns=[col for col in base.columns if col.endswith('_y')])
base.columns=base.columns.str.replace("_x","", regex=True)
drop_cols=[col for col in base.columns if col.endswith('F') or col.endswith('N')]
drop_cols_with_code=[col for col in base.columns if 'Code' in col]
base.drop(columns=drop_cols+drop_cols_with_code, inplace=True)
base.drop(columns=['Element'], inplace=True)
#print(df['Element'].unique())
base['Year']=pd.to_numeric(base['Year'],errors='coerce')
base=base.dropna(subset=['Year'])
base['Year']=base['Year'].astype(int)
#print(df.head())
#print(df['Value'].isnull().sum())
#print(df.groupby('Element')['Value'].apply(lambda x: x.isnull().sum()))
base=base.dropna()
#print(df['Value'].isnull().sum())
#print(df[df['Value'].isnull()].head())
#print(base.head())
#print(df.shape)
#print(df.info())
#print(df.isnull().sum())
encoders={}
cols=['Area','Item']
for col in cols:
    le=LabelEncoder()
    base[col]=le.fit_transform(base[col].astype(str))
    encoders[col]=le
x=base[['Area','Item','Year','pesticides_tonnes','average_rain_fall_mm_per_year','avg_temp']]
y=base['hg/ha_yield']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
lr=LinearRegression()
lr.fit(x_train,y_train)
lr_pred=lr.predict(x_test)
mae=mean_absolute_error(y_test,lr_pred)
print("Mean Absolute Error: ",mae)
rmse=root_mean_squared_error(y_test,lr_pred)
print("Root Mean Squared Error:",rmse)
r2=r2_score(y_test,lr_pred)
print("R2 Score: ",r2)
rf=RandomForestRegressor(n_estimators=50,random_state=42, max_depth=15, n_jobs=-1)
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
area=input("Enter your country: ")
area=area.strip().lower()
if area not in encoders['Area'].classes_:
    print("Country not found")
    exit()
year=int(input("Enter year: "))
crop=input("Enter crop: ").strip().lower()
pesticides=float(input("Enter pesticides used(tonnes): "))
rainfall=float(input("Enter rainfall (mm): "))
temperature=float(input("Enter temperature (°C): "))
new_data=x.iloc[:1].copy()
new_data['pesticides_tonnes']=pesticides
new_data['average_rain_fall_mm_per_year']=rainfall
new_data['avg_temp']=temperature
if 'production_tonnes' in new_data.columns:
    new_data['production_tonnes']=base['production_tonnes'].median()
if crop not in encoders['Item'].classes_:
    print("Crop not found")
    exit()
Area_Enc=encoders['Area'].transform([area])[0]
Crop_Enc=encoders['Item'].transform([crop])[0]
new_data['Area']=Area_Enc
new_data['Year']=year
new_data['Item']=Crop_Enc
prediction=rf.predict(new_data)
print("Predicted Crop Yield (hg/ha yield): ",prediction[0])
print("train",rf.score(x_train,y_train))
print("test",rf.score(x_test,y_test))
print(x.columns)
#print(base.shape)
#print(base.corr(numeric_only=True)['hg/ha_yield'].sort_values(ascending=False))
model=joblib.dump(rf,"crop_yield_predictor_model.pkl")
encoders=joblib.dump(encoders,"encoders.pkl")
base.to_csv(r"Smart_Crop_Yield_Predictor.csv",index=False)