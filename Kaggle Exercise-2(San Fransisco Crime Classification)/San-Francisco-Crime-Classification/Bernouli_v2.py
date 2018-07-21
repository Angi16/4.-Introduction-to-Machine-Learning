# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 19:06:11 2016

@author: Aditya
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:42:01 2016

@author: Aditya
"""
#Importing required libraries and packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import log_loss

#Loading the data
train = pd.read_csv("C:/Users/Aditya/Desktop/SCM-CIS 593/San Fran Project/train.csv/train.csv", parse_dates = ["Dates"], index_col= False)
test = pd.read_csv("C:/Users/Aditya/Desktop/SCM-CIS 593/San Fran Project/test.csv/test.csv",parse_dates=["Dates"], index_col = False)
train.info()
#Sampling
#sampling = 0.65

#dropping the variables Description and Resolution from test as they are not present in test
train = train.drop(["Descript", "Resolution"], axis=1)
#test= test.drop(["Address"], axis = 1)
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
#Encoding the categorical variables
cat_encoder = LabelEncoder()
cat_encoder.fit(train["Category"])
#train["Address"]= add_encoder.transform(train["Address"])
train["CategoryEncoded"]= cat_encoder.transform(train["Category"])
train = pd.concat([train,pd.get_dummies(train.PdDistrict)], axis=1)
train = pd.concat([train,pd.get_dummies(train.DayOfWeek)], axis=1)
train["Address"]= train["Address"].apply(lambda x: 1 if "/" in x else 0)
train["Dark"] = train["Hour"].apply(lambda x: 1 if x>= 19 or x <=5 else 0)
#add_encoder = LabelEncoder()
test = pd.concat([test,pd.get_dummies(test.PdDistrict)], axis=1)
test = pd.concat([test,pd.get_dummies(test.DayOfWeek)], axis=1)
test["Address"] = test["Address"].apply(lambda x: 1 if "/" in x else 0)
test["Dark"] = test["Hour"].apply(lambda x: 1 if x>=19 or x<=5 else 0)
#add_encoder.fit(test["Address"])
#test["Address"]=add_encoder.transform(test["Address"])
print(cat_encoder.classes_)
print(train.columns)
print(test.columns)
train = train.drop(["CategoryEncoded"], axis=1)
#Select only the required columns
train_columns = list(train.columns[4:].values)
print(train_columns)
test_columns = list(test.columns[4:].values)
print(test_columns)
#Split Coordinates:
#x_train = train[:int(len(train) * sampling)]
#x_test = train[int(len(train) * sampling):]
#scores=[]
#RandomForest Classifier
#classifier = RandomForestClassifier(n_estimators=75,criterion="entropy",bootstrap=True)
#classifier.fit(train[train_columns], train["CategoryEncoded"])
#scores.append(classifier.score(x_test[train_columns], x_test["CategoryEncoded"]))
#test["predictions"] = classifier.predict(test[test_columns])
#Creating the submission file
#def field_to_columns(data, field, new_columns):
#    for i in range(len(new_columns)):
#        data[new_columns[i]] = (data[field] == new_columns[i]).astype(int)
#    return data
#test["Category"]= cat_encoder.inverse_transform(test["predictions"])
#categories = list(cat_encoder.classes_)
#test = field_to_columns(test, "Category", categories)
#print(test.columns)
#PREDICTIONS_FILENAME = 'predictions_'+ '.csv'
#submission_cols = [test.columns[0]]+list(test.columns[13:])
#print(submission_cols)
#test[submission_cols].to_csv(PREDICTIONS_FILENAME, index = False)
#print(scores)
#scaler = preprocessing.StandardScaler().fit(train[train_columns])

#knn=KNeighborsClassifier(n_neighbors=23, weights='distance',algorithm='auto',metric="minkowski", p=3)

#knn.fit(scaler.transform(train[train_columns]),
#       train['Category'])
#test['hour'] = test['date'].str[11:13]
# Separate test and train set out of orignal train set.

#train['pred'] = knn.predict(scaler.transform(train[train_columns]))
#test_pred = knn.predict_proba(scaler.transform(test[test_columns]))
#Bernouli model
training,validation = train_test_split(train, train_size=.75)
model = BernoulliNB(alpha=1.0, fit_prior = True)
model.fit(train[train_columns], train["Category"])
predicted = model.predict_proba(test[test_columns])
predict = model.predict_proba(validation[train_columns])
lgloss = log_loss(validation["Category"], predict)     
print(lgloss)                                       
# EXPORT TEST SET PREDICTIONS.
# This section exports test predictions to a csv in the format specified by Kaggle.com.
result = pd.DataFrame(predicted,columns=cat_encoder.classes_)
result.to_csv("testresult1.csv", index = True, index_label = "Id")
