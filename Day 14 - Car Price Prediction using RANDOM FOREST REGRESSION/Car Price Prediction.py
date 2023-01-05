"Day 14 | Car Price Prediction using RANDOM FOREST"""

"""Import Libraries"""
import pandas as pd

"""Load Dataset from Local directory"""
print("1. Load & Summarize Dataset:")

# Load Dataset
dataset = pd.read_csv('Car Price Prediction Dataset.csv')
dataset = dataset.drop('car_ID', axis='columns')

# Summarize Dataset
print(f'    Shape of Dataset: {dataset.shape}')
print(f'    Top Head of Dataset: \n{dataset.head(5)}')

"""Splitting Dataset into X & Y"""
print("\n2. Segregate Dataset into X & Y:")

# (X_data) = This X contains Both Numerical & Text Data
X_data = dataset.drop('price', axis='columns')
numerical_X_Columns = X_data.select_dtypes(exclude=['object']).columns
X = X_data[numerical_X_Columns]

# Output = Y
Y = dataset['price']
print(f'    X = \n{X}\n     Y = \n{Y}')

"""Scaling the Independent Variables (Features)"""
print("\n3. Scaling the Independent Variables (Features):")
from sklearn.preprocessing import scale

cols = X.columns
print(cols)
X = pd.DataFrame(scale(X))
print(X)
X.columns = cols
print(X.columns)

"""Splitting Dataset into Train & Test"""
print("\n4. Splitting Dataset into train & test:")
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
print(f'    X_train = {X_train.shape}')
print(f'    X_test  =  {X_test.shape}')
print(f'    Y_train = {Y_train.shape}')
print(f'    Y_test  =  {Y_test.shape}')

"""Training using Random Forest"""
print("\n5. Prediction and Validation:")
from sklearn.ensemble import RandomForestRegressor

# Training
model = RandomForestRegressor()
model.fit(X_train.values, Y_train.values)

# Evaluating Model
"""Evaluating Model"""
Y_predicted = model.predict(X_test.values)
from sklearn.metrics import r2_score

r2score = r2_score(Y_test.values, Y_predicted)
print("     R2Score", r2score * 100)

"""Prediction"""
print("\n6. Prediction:")

Symboling = 3
Wheelbase = 88.6
CarLength = 168.8
CarWidth = 64.1
CarHeight = 48.8
CurbWeight = 2548
EngineSize = 130
BoreRatio = 3.47
Stroke = 2.68
Compression = 9
HorsePower = 111
PeakRPM = 5000
CityMPG = 21
Highwaympg = 27

newCust = [[Symboling, Wheelbase, CarLength, CarWidth, CarHeight, CurbWeight, EngineSize, BoreRatio, Stroke, Compression, HorsePower, PeakRPM, CityMPG, Highwaympg]]

Predicted_modelResult = model.predict(pd.DataFrame(scale(newCust)))
print("     Car price is: ", Predicted_modelResult[0])
