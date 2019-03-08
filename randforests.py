import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data=pd.read_csv('Fdata1.csv')
d1=pd.read_csv('Fdata.csv')
data['label'] = data['label'].map({'Standing': 1, 'Walking': 0})
x=data.ix[:,(0,1,2)].values
y=data.ix[:,3].values
x2=d1.ix[:,(0,1,2)].values


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)  

regressor = RandomForestRegressor(n_estimators=20, random_state=0)  
regressor.fit(X_train, y_train)  
y_pred = regressor.predict(X_test)
print(y_pred)
