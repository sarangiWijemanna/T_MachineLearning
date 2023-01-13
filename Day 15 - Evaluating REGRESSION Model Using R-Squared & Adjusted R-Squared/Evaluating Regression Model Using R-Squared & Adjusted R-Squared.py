"Day 15 | Evaluating Regression Model Using R-Squared & Adjusted R-Squared"""

"""Importing Libraries"""
import pandas as pd
import matplotlib.pyplot as plt

"""Load and Summarizing Dataset"""
print("1.Load and Summarizing Dataset: ")
# Load dataset from Local Directory
#   from google.colab import files
#   uploaded = files.upload()

# Load dataset
dataset = pd.read_csv("Price Prediction Dataset1.csv")

# Summarized dataset
print(f'    Shape of Dataset: {dataset.shape}')
print(f'    head(5) of Dataset: \n{dataset.head(5)}')

"""Visualizing Dataset"""
print("\n2. Visualizing Dataset:")
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(dataset.area, dataset.price, color='green', marker='*')
plt.show()

"""Segregating Dataset into X & Y"""
print("\n3.Segregating Dataset into X & Y:")

X = dataset.drop("price", axis='columns')
print(f'    X = \n{X}')
Y = dataset.price
print(f'    Y = \n{Y}')

"""Splitting Dataset for Testing our Model"""
print("\n4.Splitting Dataset for Testing our Model:")
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

"""Training - Using Liner Regression Algorithm"""
print("\n5. Training - Using Liner Regression Algorithm:")
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train.values, Y_train.values)

"""Visualizing Linear Regression results"""
print("\n6.Visualizing Linear Regression results:")

plt.scatter(X, Y, color="red", marker='*')
plt.plot(X, model.predict(X.values))
plt.title("Linear Regression")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()

"""Evaluation Model"""
print("\n7.Evaluation Model:")

""" **** R Squared = 1- (SSR/SST)
where,
    SSR = Sum of Squared Residuals

    SST = Sum of Squared Total

   **** Adjusted R Squared= 1 — [(1 — R Squared) * ((n-1) / (n-p-1))]"""

# R-Squared Score
print("\n7.1.R-Squared Score of the Model:")
r_Squared = model.score(X_test.values, Y_test.values)
print(r_Squared)

"""Adjusted R Squared of the Model"""
print("\n7.2.Adjusted R Squared of the Model:")
n = len(dataset)  # Length of Total dataset
p = len(dataset.columns) - 1  # length of Features

adjusted_R_Squared = 1 - (1 - r_Squared) * (n - 1) / (n - p - 1)
print(adjusted_R_Squared)

"""Prediction"""
print("\n8. Prediction:")

x = 6500
customer_Gives_LandArea_inSqFeet = [[x]]
predicted_Price_ModelResult = model.predict(customer_Gives_LandArea_inSqFeet)
print(f'Predicted Price by the Model: {predicted_Price_ModelResult[0]}')


