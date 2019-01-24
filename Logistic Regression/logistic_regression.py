#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 22:05:44 2019

@author: adityakhera
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


dataset = pd.read_csv("breast-cancer-wisconsin.csv", header = None)


dataset.fillna(dataset.mean() ,inplace = True)

#---Data Preprocessing---
X = dataset.iloc[:, 1:-1]
y = dataset.iloc[:, -1]




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


#Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



#Fitting Logistic Regression to training set
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

#Predict test set results
y_pred = classifier.predict(X_test)

#Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)
print(classifier.score(X_test, y_test))

