# -*- coding: utf-8 -*-
""""
Created on Sun Oct  8 00:43:21 2017 

@author: Dean Bozic
"""

# Importing the libraries
import numpy as np # library for mathematical tools
import matplotlib.pyplot as plt # library for ploting charts
import pandas as pd # library for importing datasets

# Importing the dataset
dataset = pd.read_csv('Data.csv')
# matrix of features (independent variables)
X = dataset.iloc[:, :-1].values
# independent vector
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer # library for preporcessing
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) #axis = 0 = columns
imputer = imputer.fit(X[:, 1:3]) # 
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# taking care of country data(first index)
labelencoder_X = LabelEncoder()
# encode the values
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# dependend variables have no order
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
