" Day - 18 | Regression Model Selection for Engine Energy prediction"""

"""Importing Libraries"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""Load and Summarizing Dataset"""
print("1.Load and Summarizing Dataset: ")
# Load dataset from Local Directory
#   from google.colab import files
#   uploaded = files.upload()

# Load dataset
dataset = pd.read_csv("Engine Energy Prediction Dataset.csv")

# Summarized dataset
print(f'    Shape of Dataset: {dataset.shape}')
print(f'    head(5) of Dataset: \n{dataset.head(5)}')

"""Splitting Dataset into X & Y"""
print("\n2. Segregate Dataset into X & Y:")

# X = Input
X = dataset.iloc[:, :-1].values

# Y = Output
Y = dataset.iloc[:, -1].values


"""Splitting Dataset into Train & Test"""
print("\n3. Splitting Dataset into train & test:")
from sklearn.model_selection import train_test_split

# For all algorithm
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
print(f'    X_train = {X_train.shape}')
print(f'    X_test  =  {X_test.shape}')
print(f'    Y_train = {Y_train.shape}')
print(f'    Y_test  =  {Y_test.shape}')

# For SVM
X_train_svm, X_test_svm, Y_train_svm, Y_test_svm = train_test_split(X, Y, test_size=0.20, random_state=0)
print(f'\n    X_train_svm = {X_train_svm.shape}')
print(f'    X_test_svm  =  {X_test_svm.shape}')
print(f'    Y_train_svm = {Y_train_svm.shape}')
print(f'    Y_test_svm  =  {Y_test_svm.shape}')

"""Importing Ml Algorithm"""
print("\n4. Importing Ml Algorithm:")

from sklearn.linear_model import LinearRegression
print("...imported LinearRegression...")

from sklearn.preprocessing import PolynomialFeatures
print("...imported PolynomialFeatures...")

from sklearn.ensemble import RandomForestRegressor
print("...imported RandomForestRegressor...")

from sklearn.tree import DecisionTreeRegressor
print("...imported DecisionTreeRegressor...")

from sklearn.svm import SVR
print("...imported SVR...")

"""Initialing Different Regression Algorithm"""
print("\n5. Initialing Different Regression Algorithm:")

from sklearn.preprocessing import StandardScaler

model_LR = LinearRegression()

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X_train)
model_PLR = LinearRegression()

model_RFR = RandomForestRegressor(n_estimators=10, random_state=0)

model_DTR = DecisionTreeRegressor(random_state=0)

model_SVR = SVR(kernel='rbf')

# For SVR
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_svm = sc_X.fit_transform(X_train_svm)
# X_test_svm = sc_y.fit_transform(X_test_svm)



"""Training Regression Algorithm"""
print("\n6. Training Regression Algorithm:")

model_LR.fit(X_train, Y_train)
print("...Trained using Linear Regression...")

model_PLR.fit(X_poly, Y_train)
print("...Trained using Polynomial Regression...")

model_RFR.fit(X_train, Y_train)
print("...Trained using Random Forest Regression...")

model_DTR.fit(X_train, Y_train)
print("...Trained using Decision Tree Regression...")

# Explanation:
# .values will give the values in a numpy array (shape: (n,1))
# .ravel will convert that array shape to (n, ) (i.e. flatten it)
model_SVR.fit(X_train_svm, Y_train_svm)
print("...Trained using Support Vector Regression...")

"""Predicting the Test set for Validation"""
print("\n7. Predicting the Test set for Validation:")

modelLRy_prediction = model_LR.predict(X_test)
modelPLRy_prediction = model_PLR.predict(poly_reg.transform(X_test))
modelRFRy_prediction = model_RFR.predict(X_test)
modelDTRy_prediction = model_DTR.predict(X_test)
modelSVRy_prediction = model_SVR.predict(sc_X.transform(X_test_svm))


"""Evaluating the Model Performance"""
print("\n8. Evaluating the Model Performance:")

from sklearn.metrics import r2_score
print("Linear Regression Accuracy: {}".format(r2_score(Y_test, modelLRy_prediction)))
print("Polynomial Regression Accuracy: {}".format(r2_score(Y_test, modelPLRy_prediction)))
print("Random Forest Regression Accuracy: {}".format(r2_score(Y_test, modelRFRy_prediction)))
print("Decision Tree Regression Accuracy: {}".format(r2_score(Y_test, modelDTRy_prediction)))
print("Support Vector Regression Accuracy: {}".format(r2_score(Y_test, modelSVRy_prediction)))


