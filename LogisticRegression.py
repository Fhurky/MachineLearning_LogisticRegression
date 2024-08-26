# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:34:55 2024

@author: furko
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, jaccard_score, log_loss


data = pd.read_csv("Social_Network_Ads.csv")

data = data.drop(["User ID"], axis = 1)
data["Gender"] = pd.factorize(data["Gender"])[0] + 1


scaler = StandardScaler()
scaler.fit(data)
data_scaled = scaler.transform(data)

data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

Y = data.iloc[:,-1]
X = data_scaled.drop(["Purchased"], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=53)

LR = LogisticRegression(solver="liblinear", C=0.0001)

LR.fit(x_train, y_train)

LR.predict(x_test)

print ("LogLoss: : %.2f" % log_loss(y_test, LR.predict_proba(x_test)))

print ("Jaccard_score: : %.2f" % jaccard_score(y_test, LR.predict(x_test),pos_label=0))