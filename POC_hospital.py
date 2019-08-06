#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 17:39:38 2019

@author: angadsingh
"""

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('clevelanda.csv')

# data understanding
df.info()
df.describe()
df['ca'].value_counts()
df['thal'].value_counts()
df.isnull().sum()
df.columns
df.hist()
df.boxplot()
df.head()


# data preprocessing

df = df.replace(['?'],[np.nan])

df.isnull().sum()

df = df.fillna(df.mean())

df = df.astype(float)

#converting classes - clients only wants 0 and 1 classes
d = df.groupby(df.iloc[:,13]).mean()
 
X = df.iloc[:,:13]
y = df.iloc[:,13]

y = y.replace([4],[1])
y = y.replace([3],[1])
y = y.replace([2],[1])
 
#splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
 
# feature selection of model
from sklearn.feature_selection import RFE
#logreg = DecisionTreeClassifier()
logreg = LogisticRegression()
#logreg = RandomForestClassifier()
rfe = RFE(logreg, 8)
rfe.fit(X_train, y_train)

print(rfe.support_)
print(rfe.ranking_)

#model = DecisionTreeClassifier()
#model = RandomForestClassifier()
model = LogisticRegression()
model.fit(X_train, y_train)
print(model)
expected = y_test
pred = model.predict(X_test)
print(metrics.classification_report(expected, pred))
print(confusion_matrix(expected,pred))
 
 
 
 
 
 
 
 
 