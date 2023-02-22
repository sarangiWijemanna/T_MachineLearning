# Day-6 | Digit Recognition using RANDOM FOREST

# Import Libraries
import pandas as pd
import numpy as np

""" ## Load and Summarized Dataset"""
print("1. Load and Summarized Dataset: ")

# Access Google Drive contents*"""
#    from google.colab import drive
#    drive.mount('/content/gdrive')

# Load Dataset
fileName = "Digit Recognition Dataset.csv"
dataset = pd.read_csv(fileName)
print(dataset)

# Summarized Dataset
print(dataset.shape)
print(dataset.head(5))

"""## Segregate Dataset into X & Y """
print("\n2. Segregate Dataset into X & Y: ")
# X = (Input/IndependentVariable)
X = dataset.iloc[:, 1:]
print(f'X = \n{X}')
print(f'Shape of X = {X.shape}')

# Y = (Output/DependentVariable)
Y = dataset.iloc[:, 0]
print(f'\nY = \n{Y}')
print(f'Shape of Y = {Y.shape}')

""" ## Splitting Dataset into Train (75%) and Test(25%)"""
print("\n3. Splitting Dataset into Train & Test:")
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
print(X_train.shape)

"""## Training : Trained"""
print("\n4. Training: ")
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, Y_train)

# Y_prediction when put X_test Data
Y_prediction = model.predict(X_test)

"""## Accuracy Score """
from sklearn.metrics import accuracy_score

print("\n5.  Accuracy of the Model: {0}%".format(accuracy_score(Y_test, Y_prediction) * 100))

"""## Prediction """
print("\n6. Prediction: ")
import matplotlib.pyplot as plt

index = 1

model_Predicted_Value = model.predict(X_test)[index]
print("Predicted " + str(model_Predicted_Value))
plt.title("Result: "'%i' % model_Predicted_Value)
plt.axis('off')
plt.imshow(X_test.iloc[index].values.reshape((28, 28)), cmap='gray')

# plt.show()

"""## Try with the Changing HyperParameter """
print("\n\n7. Try with the Changing HyperParameter: ")
from sklearn.ensemble import RandomForestClassifier

model_1 = RandomForestClassifier()
model_2 = RandomForestClassifier(n_estimators=200)
model_3 = RandomForestClassifier(n_estimators=300, criterion='gini', max_depth=15,  max_features=50, random_state=0)
model_4 = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=25,  bootstrap=False, random_state=0)

model_1.fit(X_train, Y_train)
model_2.fit(X_train, Y_train)
model_3.fit(X_train, Y_train)
model_4.fit(X_train, Y_train)

Y_prediction_model_1 = model_1.predict(X_test)
Y_prediction_model_2 = model_2.predict(X_test)
Y_prediction_model_3 = model_3.predict(X_test)
Y_prediction_model_4 = model_4.predict(X_test)

print("Accuracy of the Model 1: {0}%".format(accuracy_score(Y_test, Y_prediction_model_1) * 100))
print("Accuracy of the Model 2: {0}%".format(accuracy_score(Y_test, Y_prediction_model_2) * 100))
print("Accuracy of the Model 3: {0}%".format(accuracy_score(Y_test, Y_prediction_model_3) * 100))
print("Accuracy of the Model 3: {0}%".format(accuracy_score(Y_test, Y_prediction_model_4) * 100))
