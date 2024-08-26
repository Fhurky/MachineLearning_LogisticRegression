# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:34:55 2024

@author: furko
"""

import pandas as pd

data = pd.read_csv("Social_Network_Ads.csv")

data = data.drop(["User ID"], axis = 1)