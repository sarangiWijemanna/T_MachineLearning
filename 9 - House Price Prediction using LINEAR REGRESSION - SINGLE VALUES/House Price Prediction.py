"Day 9| House price prediction using Linear Regression-Single Variable"""

"""Importing Libraries"""
import pandas as pd
import matplotlib.pyplot as plt

"""Load and Summarizing Dataset"""
print("1.Load and Summarizing Dataset: ")
# Load dataset from Local Directory
#   from google.colab import files
#   uploaded = files.upload()

# Load dataset
dataset = pd.read_csv("House Price Prediction Dataset.csv")

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

"""Training - Using Liner Regression Algorithm"""
print("\n4. Training - Using Liner Regression Algorithm:")
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X.values, Y.values)

"""Prediction"""
print("\n5. Prediction:")

x = 2000
customerGivesLandArea_inSqFeet = [[x]]
predictedPrice_ModelResult = model.predict(customerGivesLandArea_inSqFeet)
print(f'Predicted Price by the Model: {predictedPrice_ModelResult}')

# Check Out Model is Right?
#   Y = mx + C
print("\n Check Out Model is Right?:")
m = model.coef_
print(f'    m = {m}')
C = model.intercept_
print(f'    C = {C}')

y = m * x + C
print("\nPrice of {0} Sq.Feet of the Land: {1}".format(x, y[0]))
print("\nPrice of {0} Sq.Feet of the Land: {1}".format(x, y))
