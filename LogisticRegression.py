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

# Load the dataset
data = pd.read_csv("Social_Network_Ads.csv") 

# Drop the "User ID" column as it is not useful for the model
data = data.drop(["User ID"], axis=1)

# Convert the "Gender" column to numerical values
data["Gender"] = pd.factorize(data["Gender"])[0] + 1

# Initialize the StandardScaler and fit it to the data
scaler = StandardScaler()
scaler.fit(data)

# Scale the data using the fitted scaler
data_scaled = scaler.transform(data)

# Convert the scaled data back into a DataFrame
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

# Define the target variable Y and the features X
Y = data.iloc[:,-1] # Y value must be non-continious
X = data_scaled.drop(["Purchased"], axis=1)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Initialize the Logistic Regression model with specific parameters
LR = LogisticRegression(solver="liblinear", C=0.0001)

# Train the Logistic Regression model
LR.fit(x_train, y_train)

# Calculate and print the Log Loss metric
print("LogLoss: : %.2f" % log_loss(y_test, LR.predict_proba(x_test)))

# Calculate and print the Jaccard Score for the test set
print("Jaccard_score: : %.2f" % jaccard_score(y_test, LR.predict(x_test), pos_label=0))

# Generate and print the Confusion Matrix
print("Confusion_Matrix:\n", confusion_matrix(y_test, LR.predict(x_test), labels=[1,0]))
