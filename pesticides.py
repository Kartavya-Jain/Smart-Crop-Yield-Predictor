import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
pesticides=pd.read_csv(r"D:\Projects\Smart Crop Yield Predictor\Dataset\pesticides.csv")
#print(pesticides.head())
#pesticides.info()
#print(pesticides.isnull().sum())
#print(pesticides.duplicated().sum())
#print(pesticides.head())
#print(pesticides.columns)
#cols=['Domain', 'Area', 'Element', 'Item', 'Year', 'Unit', 'Value']
#for x in cols:
#   print(pesticides[x].nunique()) 
#No need to clean Domain as it has only one unique value
pesticides['Area']=pesticides['Area'].str.strip().str.replace(r'\s+',' ',regex=True)
#print(pesticides['Area'])
#No need to clean Element as it has only one unique value
#No need to clean Item as it has only one unique value
#No need to clean Year as it has only integer value
#No need to clean Unit as it has only one unique value
#No need to clean Value as the data is in float
#sns.kdeplot(pesticides['Value'])
#plt.show()
#print(pesticides.describe())
#print(pesticides[['Area','Value']].sort_values('Value',ascending=False).head(10))
#print(pesticides['Value'])
pesticides.drop(['Domain','Element','Unit'],axis=1,inplace=True)
le=LabelEncoder()
pesticides['Area_Enc']=le.fit_transform(pesticides['Area'])
#pesticides.drop(['Area'],axis=1,inplace=True)
print(pesticides.head())
corr_with_values=pesticides.corr(numeric_only=True)['Value'].sort_values(ascending=False)
#print(corr_with_values)
X=pesticides[['Area_Enc','Year']]
y=pesticides['Value']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
lr=LinearRegression()
lr.fit(X_train,y_train)
lr_pred=lr.predict(X_test)
#mae=mean_absolute_error(y_test,lr_pred)
#print("Mean Absolute Error: ",mae)
#rmse=root_mean_squared_error(y_test,lr_pred)
#print("Root Mean Squared Error: ",rmse)
#r2=r2_score(y_test,lr_pred)
#print("R2 Score: ",r2)
rf=RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(X_train,y_train)
rf_pred=rf.predict(X_test)
r2=r2_score(y_test,rf_pred)
print("R2 Score: ",r2)
print("Train score:", rf.score(X_train, y_train))
print("Test score:", rf.score(X_test, y_test))
#for i,label in enumerate(le.classes_):
#   print(i,label)
#print(pesticides['Year'].min(),pesticides['Year'].max())#1990 2016
area=input("Enter your country: ")
year=int(input("Enter year: "))
Area_Enc=le.transform([area])[0]
new_data=pd.DataFrame({
   'Area_Enc':[Area_Enc],
   'Year':[year]
})
prediction=rf.predict(new_data)
print("Predicted pesticide value: ",prediction[0])
#pesticides.to_csv(r"C:\Users\Karta\OneDrive\Desktop\Smart Crop Yield Predictor\Dataset\pesticides_cleaned.csv",index=False)     