import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors,cross_validation,preprocessing

d1=pd.read_csv('Fdata1.csv')
d2=pd.read_csv('Fdata.csv')
X=d1.ix[:,(0,1,2)].values
y=d1.ix[:,(3)].values
x2=d2.ix[:,(0,1,2)].values

X_train, X_test, y_train, y_test =cross_validation.train_test_split(X, y, test_size=0.20)
classifier = neighbors.KNeighborsClassifier()  
classifier.fit(X_train, y_train)
accuracy=classifier.score(X_test,y_test)
y_pred = classifier.predict(x2)
print(accuracy)
#print(y_pred)
print(y_pred[0])
