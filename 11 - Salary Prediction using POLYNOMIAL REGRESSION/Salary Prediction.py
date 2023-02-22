"Day 11 | Salary Prediction using POLYNOMIAL REGRESSION"""

"""Import Libraries"""
import pandas as pd
import matplotlib.pyplot as plt

"""Load and Summarize Dataset"""
print('1. Load & Summarize Dataset: ')
# Load dataset
dataset = pd.read_csv("Salary Prediction Dataset.csv")

# Summarize Dataset
print(f'    Shape of Dataset: {dataset.shape}')
print(f'    Top Head of Dataset: \n{dataset.head(5)}')

"""Segregate Dataset into X & Y"""
print('\n2. Segregate Dataset into X & Y:')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
print(f'    X = \n{X}\n     Y = \n{Y}')

"""Training Using Linear Regression"""
print('\n3. Training - Linear Regression:')

# Step 1 - Training using LR
from sklearn.linear_model import LinearRegression

model_LR = LinearRegression()
model_LR.fit(X, Y)

# Step 2 - Visualizing Linear Regression Results
plt.scatter(X, Y, color='green')
plt.plot(X, model_LR.predict(X))
plt.title("Linear Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

"""Training Using Polynomial Regression"""
print('\n4. Training - Polynomial Regression:')

# Step 1 - Convert Normal Features X into Polynomial format
from sklearn.preprocessing import PolynomialFeatures

model_Poly_R = PolynomialFeatures(degree=4)
xPloy = model_Poly_R.fit_transform(X)
# print(f'    X = \n{xPloy}')

# Step 2 - Training LR using Polynomial format feature (X --> X^n)
model_Poly_LR = LinearRegression()
model_Poly_LR.fit(xPloy, Y)

# Step 3 - Visualizing Polynomial Regression Results
plt.scatter(X, Y, color='green')
plt.plot(X, model_Poly_LR.predict(model_Poly_R.fit_transform(X)))
plt.title("Polynomial Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

"""Prediction Using Polynomial Regression"""
print('\n5. Prediction Using Polynomial Regression:')

Level_Employee = 8.5
salary_Predicted = model_Poly_LR.predict(model_Poly_R.fit_transform([[Level_Employee]]))
print('Salary of a New Employee with Level {0} is {1} $'.format(Level_Employee, salary_Predicted[0]))
