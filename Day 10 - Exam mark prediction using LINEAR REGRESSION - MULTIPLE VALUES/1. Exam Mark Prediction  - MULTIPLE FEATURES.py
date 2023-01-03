"Day 10 | Exam mark prediction using Linear Regression-Multiple Variable"""

"""Importing Libraries"""
import pandas as pd

"""Load & Summarize Dataset"""
print('1. Load & Summarize Dataset: ')
# Load dataset
dataset = pd.read_csv("Exam Mark Dataset.csv")

# Summarize dataset
print(f'    Shape of Dataset: {dataset.shape}')
print(f'    Top Head of Dataset: \n{dataset.head(5)}')

"""Find & Remove NA Values (Empty Cells with mean of relevant feature"""
print('\n2. Find & Remove NA Values:')
print(f'    NA Existed in : {dataset.columns[dataset.isna().any()]}')
dataset.hours = dataset.hours.fillna(dataset.hours.mean())
print(dataset.head(5))

"""Segregate Dataset into X & Y"""
print('\n3. Segregate Dataset into X & Y:')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
print(f'    X = \n{X}\n     Y = \n{Y}')

"""Training - Linear Regression"""
print('\n4. Training - Linear Regression:')
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, Y)

"""Prediction for Customer values"""
print('\n5. Prediction for Customer values:')
landArea_inSqFt = [[8.67, 19, 1]]
predictedModelResult = model.predict(landArea_inSqFt)
print(f'Predicted Model Result: {predictedModelResult}')
