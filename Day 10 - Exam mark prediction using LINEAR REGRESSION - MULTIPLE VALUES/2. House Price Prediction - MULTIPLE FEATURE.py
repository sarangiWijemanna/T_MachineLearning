"Day 10 | House price prediction using Linear Regression-Multiple Variable"""

"""Importing Libraries"""
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

"""Load & Summarize Dataset"""
print('1. Load & Summarize Dataset: ')
# Load dataset
dataset = pd.read_csv("House Price Prediction Dataset.csv")

# Summarize dataset
print(f'    Shape of Dataset: {dataset.shape}')
print(f'    Top Head of Dataset: \n{dataset.head(5)}')

print("     MSZoning: {0}".format(dataset['MSZoning'].unique()))
print("     HouseStyle: {0}".format(dataset['HouseStyle'].unique()))

"""Find & Remove NA Values (Empty Cells with mean of relevant feature"""
print('\n2. Find & Remove NA Values:')
print(f'    NA Existed in : {dataset.columns[dataset.isna().any()]}')
dataset['LotFrontage'] = dataset['LotFrontage'].fillna(dataset['LotFrontage'].mean())

"""Convert string to Numerical values"""
print('\n3. Convert string to Numerical values:')

le = LabelEncoder()
dataset['MSZoning'] = le.fit_transform(dataset['MSZoning'])
print(f"   MSZoning : \n{dataset['MSZoning']}")
dataset['HouseStyle'] = le.fit_transform(dataset['HouseStyle'])
print(f"   HouseStyle : \n{dataset['HouseStyle']}")

"""Segregate Dataset into X & Y"""
print('\n4. Segregate Dataset into X & Y:')
X = dataset[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'HouseStyle', 'OverallQual', 'OverallCond']].values
Y = dataset['SalePrice'].values
print(f'    X = \n{X}\n     Y = \n{Y}')

"""Training - Linear Regression"""
print('\n4. Training - Linear Regression')

model = LinearRegression()
model.fit(X, Y)

"""Prediction for Customer values"""
print('\n5. Prediction for Customer values:')

MSSubClass = 1100
MSZoning = 3
LotFrontage = 65
LotArea = 5
HouseStyle = 5
OverallQuality = 7
OverallCond = 5

housePricePredictionFeatures = [[MSSubClass, MSZoning, LotFrontage, LotArea, HouseStyle, OverallQuality, OverallCond]]
ModelPredictedResult = model.predict(housePricePredictionFeatures)
print(" Predicted House Price: {0} $ ".format(ModelPredictedResult[0]))
