# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:42:01 2016

@author: Aditya
"""
#Importing required libraries and packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier 

#Loading the data
train = pd.read_csv("C:/Users/Aditya/Desktop/SCM-CIS 593/San Fran Project/train.csv/train.csv", parse_dates = ["Dates"], index_col= False)
test = pd.read_csv("C:/Users/Aditya/Desktop/SCM-CIS 593/San Fran Project/test.csv/test.csv",parse_dates=["Dates"], index_col = False)
train.info()

#dropping the variables Description and Resolution from test as they are not present in test
train = train.drop(["Descript", "Resolution","Address"], axis=1)
test= test.drop(["Address"], axis = 1)
train.info()

#Splitting the date into year, time
def datesplit(data):
    data["Year"] = data["Dates"].dt.year
    data["Month"] = data["Dates"].dt.month
    data["Day"] = data["Dates"].dt.day
    data["Hour"] = data["Dates"].dt.hour
    data["Minute"] = data["Dates"].dt.minute
    return data
train = datesplit(train)
test = datesplit(test)
enc = LabelEncoder()
train["PdDistrict"] = enc.fit_transform(train["PdDistrict"])
wnc = LabelEncoder()
train["DayOfWeek"] = wnc.fit_transform(train["DayOfWeek"])
cat_encoder = LabelEncoder()
cat_encoder.fit(train["Category"])
train["CategoryEncoded"]= cat_encoder.transform(train["Category"])
print(cat_encoder.classes_)
enc = LabelEncoder()
test["PdDistrict"]= enc.fit_transform(test["PdDistrict"])
wnc = LabelEncoder()
test["DayOfWeek"] = wnc.fit_transform(test["DayOfWeek"])
print(train.columns)
print(test.columns)
train_columns = list(train.columns[2:11].values)
print(train_columns)
test_columns = list(test.columns[2:11].values)
print(test_columns)
classifier = RandomForestClassifier(n_estimators=15,criterion="entropy",bootstrap=True)
classifier.fit(train[train_columns], train["CategoryEncoded"])
test["predictions"] = classifier.predict(test[test_columns])
def field_to_columns(data, field, new_columns):
    for i in range(len(new_columns)):
        data[new_columns[i]] = (data[field] == new_columns[i]).astype(int)
    return data
test["Category"]= cat_encoder.inverse_transform(test["predictions"])
categories = list(cat_encoder.classes_)
test = field_to_columns(test, "Category", categories)
print(test.columns)
PREDICTIONS_FILENAME = 'predictions_'+ '.csv'
submission_cols = [test.columns[0]]+list(test.columns[13:])
print(submission_cols)
test[submission_cols].to_csv(PREDICTIONS_FILENAME, index = False)


    
