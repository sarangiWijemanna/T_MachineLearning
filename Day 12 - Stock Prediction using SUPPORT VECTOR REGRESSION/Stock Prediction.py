"Day 14 | Stock Prediction using SUPPORT VECTOR REGRESSION"""

"""Import Libraries"""
import pandas as pd
import numpy as np

"""Load & Summarize Dataset"""
print("1. Load & Summarize Dataset:")

# Load dataset
dataset = pd.read_csv("Stock Prediction Dataset.csv")

# Summarize dataset
print(f'    Shape of Dataset: {dataset.shape}')
print(f'    Top Head of Dataset: \n{dataset.head(5)}')

"""Segregate Dataset into X & Y"""
print("\n2. Segregate Dataset into X & Y:")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
print(f'    X = \n{X}\n     Y = \n{Y}')

"""Splitting Dataset into Train & Test"""
print("\n3. Splitting Dataset into Train & Test:")
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=0)
print(f'    X_train = {X_train.shape}')
print(f'    X_test  =  {X_test.shape}')
print(f'    Y_train = {Y_train.shape}')
print(f'    Y_test  =  {Y_test.shape}')

"""Training - Support Vector Regression"""
print("\n4. Training - Support Vector Regression")
from sklearn.svm import SVR

model = SVR()
model.fit(X_train, Y_train)

"""Validation Model"""
print("\n5. Validation Model:")
from sklearn.metrics import mean_squared_error, r2_score

# Predicted Y values
Y_predicted = model.predict(X_test)
# Find Root Mean Square Error
mean_Squared_Error = mean_squared_error(Y_test, Y_predicted)
print("     Mean Squared Error : ", mean_Squared_Error)
print("     Root Mean Squared Error : ", np.sqrt(mean_Squared_Error))
# Find r2_score
r2_Score = r2_score(Y_test, Y_predicted)
print("     R2Score", r2_Score * 100)

"""Prediction Using Support Vector Regression"""
print('\n6. Prediction Using Support Vector Regression:')
x = 168.181818
salary_Predicted = model.predict([[x]])
print('     Stock Price Prediction: {0} $'.format(salary_Predicted[0]))

"""Tuning Parameters Support Vector Regression"""
print('\n7. Tuning Parameters Support Vector Regression:')
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# test_size=0.10
model_1 = SVR(kernel='rbf')
model_2 = SVR(kernel='linear')
model_3 = SVR(kernel='linear', gamma='auto', epsilon=2)
model_4 = SVR(kernel='linear', C=0.4)
model_5 = SVR(kernel='linear', C=2)
model_6 = SVR(kernel='linear', C=2, epsilon=0.2, degree=4)

model_1.fit(X_train, Y_train)
model_2.fit(X_train, Y_train)
model_3.fit(X_train, Y_train)
model_4.fit(X_train, Y_train)
model_5.fit(X_train, Y_train)
model_6.fit(X_train, Y_train)

# Predicted Y values
y_prediction_Model_1 = model_1.predict(X_test)
y_prediction_Model_2 = model_2.predict(X_test)
y_prediction_Model_3 = model_3.predict(X_test)
y_prediction_Model_4 = model_4.predict(X_test)
y_prediction_Model_5 = model_5.predict(X_test)
y_prediction_Model_6 = model_6.predict(X_test)

# Find Root Mean Square Error
mean_Squared_Error_1 = mean_squared_error(Y_test, y_prediction_Model_1)
mean_Squared_Error_2 = mean_squared_error(Y_test, y_prediction_Model_2)
mean_Squared_Error_3 = mean_squared_error(Y_test, y_prediction_Model_3)
mean_Squared_Error_4 = mean_squared_error(Y_test, y_prediction_Model_4)
mean_Squared_Error_5 = mean_squared_error(Y_test, y_prediction_Model_5)
mean_Squared_Error_6 = mean_squared_error(Y_test, y_prediction_Model_6)

print("     Root Mean Squared Error 1 : ", np.sqrt(mean_Squared_Error_1))
print("     Root Mean Squared Error 2 : ", np.sqrt(mean_Squared_Error_2))
print("     Root Mean Squared Error 3 : ", np.sqrt(mean_Squared_Error_3))
print("     Root Mean Squared Error 4 : ", np.sqrt(mean_Squared_Error_4))
print("     Root Mean Squared Error 5 : ", np.sqrt(mean_Squared_Error_5))
print("     Root Mean Squared Error 6 : ", np.sqrt(mean_Squared_Error_6))

# Find r2_score
r2_Score_1 = r2_score(Y_test, y_prediction_Model_1)
r2_Score_2 = r2_score(Y_test, y_prediction_Model_2)
r2_Score_3 = r2_score(Y_test, y_prediction_Model_3)
r2_Score_4 = r2_score(Y_test, y_prediction_Model_4)
r2_Score_5 = r2_score(Y_test, y_prediction_Model_5)
r2_Score_6 = r2_score(Y_test, y_prediction_Model_6)

print("     R2 Score 1 : ", r2_Score_1 * 100)
print("     R2 Score 2 : ", r2_Score_2 * 100)
print("     R2 Score 3 : ", r2_Score_3 * 100)
print("     R2 Score 4 : ", r2_Score_4 * 100)
print("     R2 Score 5 : ", r2_Score_5 * 100)
print("     R2 Score 6 : ", r2_Score_6 * 100)
