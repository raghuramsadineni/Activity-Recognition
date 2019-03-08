
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

data=pd.read_csv('Fdata1.csv')
d1=pd.read_csv('Fdata.csv')
x=data.ix[:,(0,1,2)].values
y=data.ix[:,3].values
x2=d1.ix[:,(0,1,2)].values


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

# Creating the classifier object
clf_gini = DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=3, min_samples_leaf=1)

# Performing training
clf_gini.fit(X_train, y_train)

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth = 3, min_samples_leaf = 1)

# Performing training
clf_entropy.fit(X_train, y_train)

y_pred = clf_entropy.predict(x2)
print("Predicted values:")
print(y_pred[0])




