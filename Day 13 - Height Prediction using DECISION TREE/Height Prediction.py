"Day 13 | Height Prediction using DECISION TREE"""

"""Importing Libraries"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""Load and Summarize Dataset"""
print("1. Load & Summarize Dataset:")

# Load dataset
dataset = pd.read_csv("Height Prediction Dataset.csv")

# Summarize Dataset
print(f'    Shape of Dataset: {dataset.shape}')
print(f'    Top Head of Dataset: \n{dataset.head(5)}')

"""Segregate Dataset into X & Y"""
print("\n2. Segregate Dataset into X & Y:")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
print(f'    X = \n{X}\n     Y = \n{Y}')

"""Splitting Dataset into train & test"""
print("\n3. Splitting Dataset into train & test:")
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
print(f'    X_train = {X_train.shape}')
print(f'    X_test  =  {X_test.shape}')
print(f'    Y_train = {Y_train.shape}')
print(f'    Y_test  =  {Y_test.shape}')

"""Training: Decision Tree Regressor"""
print("\n4. Training: Decision Tree Regressor")
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()
model.fit(X_train, Y_train)

"""Visualizing Graph"""
print("\n5. Prediction and Validation:")

# X-axis using training dataset
x_val = np.arange(min(X_train), max(X_train), 0.01)
print("Arrange x_val : ", x_val)
x_val = x_val.reshape(len(x_val), 1)
print("Reshape x_val : \n", x_val)

# Data Points in Training dataset
plt.scatter(X_train, Y_train, color='green')
# Predicting Y axis for x_values
plt.plot(x_val, model.predict(x_val), color='red')

plt.title("Height Prediction using Decision Tree Regressor")
plt.xlabel('Age')
plt.ylabel('Height')
plt.show()

"""Prediction and Validation"""
print("\n6. Prediction and Validation:")
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

"""Prediction"""
print("\n6. Prediction:")
age = 10
height_Predicted = model.predict([[age]])
print('     Height Prediction for {0} years old : {1} cm'.format(age, height_Predicted[0]))
