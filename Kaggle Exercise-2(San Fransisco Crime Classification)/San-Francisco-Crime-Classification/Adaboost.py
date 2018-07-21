"""
Created on Mon Mar  7 15:42:01 2016

@author: Aditya
"""
#Importing required libraries and packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss

#Loading the data
train = pd.read_csv("C:/Users/Aditya/Desktop/SCM-CIS 593/San Fran Project/train.csv/train.csv", parse_dates = ["Dates"], index_col= False)
test = pd.read_csv("C:/Users/Aditya/Desktop/SCM-CIS 593/San Fran Project/test.csv/test.csv",parse_dates=["Dates"], index_col = False)
train.info()

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
add_encoder = LabelEncoder()
cat_encoder.fit(train["Category"])
train["CategoryEncoded"]= cat_encoder.transform(train["Category"])
train = pd.concat([train,pd.get_dummies(train.PdDistrict)], axis=1)
train = pd.concat([train,pd.get_dummies(train.DayOfWeek)], axis=1)
train['StreetNo'] = train['Address'].apply(lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)
train["Intersection"]= train["Address"].apply(lambda x: 1 if "/" in x else 0)
train['Address'] = train['Address'].apply(lambda x: x.split(' ', 1)[1] if x.split(' ', 1)[0].isdigit() else x)
add_encoder.fit(train["Address"])
train["Address"]= add_encoder.transform(train["Address"])
#train["Address"]= train["Address"].apply(lambda x: 1 if "/" in x else 0)
train["Morning"] = train["Hour"].apply(lambda x: 1 if x>= 6 or x < 12 else 0)
train["Noon"] = train["Hour"].apply(lambda x: 1 if x>= 12 or x < 17 else 0)
train["Evening"] = train["Hour"].apply(lambda x: 1 if x>= 17 or x < 20 else 0)
train["Night"] = train["Hour"].apply(lambda x: 1 if x >= 20 or x < 6 else 0)
train["Fall"] = train["Month"].apply(lambda x: 1 if x>=3 and x <=5 else 0)
train["Winter"] = train["Month"].apply(lambda x: 1 if x>=6 and x <=8 else 0)
train["Spring"] = train["Month"].apply(lambda x: 1 if x>=9 and x <=11 else 0)
train["Summer"] = train["Month"].apply(lambda x: 1 if x>=12 and x <=2 else 0)
add_encoder = LabelEncoder()
test = pd.concat([test,pd.get_dummies(test.PdDistrict)], axis=1)
test = pd.concat([test,pd.get_dummies(test.DayOfWeek)], axis=1)
test['StreetNo'] = test['Address'].apply(lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)
test["Intersection"] = test["Address"].apply(lambda x: 1 if "/" in x else 0)
test['Address'] = test['Address'].apply(lambda x: x.split(' ', 1)[1] if x.split(' ', 1)[0].isdigit() else x)
add_encoder.fit(test["Address"])
test["Address"]= add_encoder.transform(test["Address"])
#test["Address"] = test["Address"].apply(lambda x: 1 if "/" in x else 0)
#test["Dark"] = test["Hour"].apply(lambda x: 1 if x>=18 or x<=5 else 0)
test["Morning"] = test["Hour"].apply(lambda x: 1 if x>= 6 or x < 12 else 0)
test["Noon"] = test["Hour"].apply(lambda x: 1 if x>= 12 or x < 17 else 0)
test["Evening"] = test["Hour"].apply(lambda x: 1 if x>= 17 or x < 20 else 0)
test["Night"] = test["Hour"].apply(lambda x: 1 if x >= 20 or x < 6 else 0)
test["Fall"] = test["Month"].apply(lambda x: 1 if x>=3 and x <=5 else 0)
test["Winter"] = test["Month"].apply(lambda x: 1 if x>=6 and x <=8 else 0)
test["Spring"] = test["Month"].apply(lambda x: 1 if x>=9 and x <=11 else 0)
test["Summer"] = test["Month"].apply(lambda x: 1 if x>=12 and x <=2 else 0)
#add_encoder.fit(test["Address"])
#test["Address"]=add_encoder.transform(test["Address"])
train=train[abs(train["Y"])<100]
xy_scaler = StandardScaler()
xy_scaler.fit(train[["X", "Y"]])
train[["X","Y"]] = xy_scaler.transform(train[["X","Y"]])
train["rot45_X"] = 0.707 * train["X"] + 0.707 * train["Y"]
train["rot45_Y"] = 0.707 * train["Y"] - 0.707 * train ["X"]
train["rot30_X"] = 0.866 * train["X"] + 0.5 * train["Y"]
train["rot30_Y"] = 0.866 * train["Y"] - 0.5 * train["X"]
train["rot60_X"] = 0.5 * train["X"] + 0.866 * train["Y"]
train["rot60_Y"] = 0.5 * train["Y"] - 0.866 * train["X"]
train["radial_r"] = np.sqrt(np.power(train["Y"],2)+ np.power(train["X"],2))
test=test[abs(test["Y"])<100]
xy_scaler = StandardScaler()
xy_scaler.fit(test[["X", "Y"]])
test[["X","Y"]] = xy_scaler.transform(test[["X","Y"]])
test["rot45_X"] = 0.707 * test["X"] + 0.707 * test["Y"]
test["rot45_Y"] = 0.707 * test["Y"] - 0.707 * test ["X"]
test["rot30_X"] = 0.866 * test["X"] + 0.5 * test["Y"]
test["rot30_Y"] = 0.866 * test["Y"] - 0.5 * test["X"]
test["rot60_X"] = 0.5 * test["X"] + 0.866 * test["Y"]
test["rot60_Y"] = 0.5 * test["Y"] - 0.866 * test["X"]
test["radial_r"] = np.sqrt(np.power(test["Y"],2)+ np.power(test["X"],2))


print(cat_encoder.classes_)
print(train.columns)
print(test.columns)

#train = train.drop(["CategoryEncoded", "Address","StreetNo"], axis=1)
train = train.drop(["CategoryEncoded","X","Y"], axis=1)
test = test.drop(["X","Y"],axis=1)
#test = test.drop(["Address", "StreetNo"], axis = 1)
    
#Select only the required columns
train_columns = list(train.columns[4:].values)
print(train_columns)
test_columns = list(test.columns[4:].values)
print(test_columns)

#Adaboosting
training,validation = train_test_split(train, train_size=.75)
model = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 10), n_estimators = 60, learning_rate = 1.0, algorithm = "SAMME")
model.fit(train[train_columns], train["Category"])     
predicted = model.predict_proba(test[test_columns])
predict = model.predict_proba(validation[train_columns])
lgloss = log_loss(validation["Category"], predict)     
print(lgloss)                                       
# EXPORT TEST SET PREDICTIONS.
# This section exports test predictions to a csv in the format specified by Kaggle.com.
result = pd.DataFrame(predicted,columns=cat_encoder.classes_)
result.to_csv("testresult1.csv", index = True, index_label = "Id")
