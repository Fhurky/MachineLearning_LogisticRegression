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