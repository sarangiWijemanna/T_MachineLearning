# Day 2.2 -  Advertisement Sale Estimation | K-NEAREST NEIGHBOUR model

# Importing Libraries
import pandas as pd  # Performed the Loading Dataset
import numpy as np  # Performed the Array Operations
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split  # Splitting Dataset into Train & Test
from sklearn.preprocessing import StandardScaler  # Feature Scaling
from sklearn.neighbors import KNeighborsClassifier  # Find K Value and Training Model
from sklearn.metrics import confusion_matrix, accuracy_score    # Evaluating Model - CONFUSION MATRIX


print("Day 4.2 -  Advertisement Sale Estimation | K-NEAREST NEIGHBOUR model")

"""## Load and Summered the Dataset """
print("\n1. Load and Summered the Dataset: ")
# Upload Dataset from Local Machine to Colab
# from google.colab import files
# uploaded = files.upload()

# Load Dataset
dataset = pd.read_csv("Advertisement Sale Dataset.csv")

# Summarized Dataset
print(dataset.shape)
print(dataset.head(5))

""" ## Segregate Dataset into X and Y """
print("\n3. Segregate Dataset into X and Y:")
#  X --> input/ Independent Variable <- age, salary
#  Y --> output/ Dependent Variable <- status

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
print(f'X is: \n{X}')
print(f'\nY is: \n{Y}')

""" ## Splitting Dataset into Train & Test"""
print("\n4. Splitting Dataset into Train & Test:")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
print(f'Rows of X: {X.shape}')
print(f' X_train: {X_train.shape}')
print(f' X_test: {X_test.shape}')
print(f'Rows of Y: {Y.shape}')
print(f' Y_train: {Y_train.shape}')
print(f' Y_test: {Y_test.shape}')

"""## Feature Scaling"""
# we scale data to make all the features contribute equally to the final result
# Fit_Transform - fit method is calculating the mean and variance of each of the features present in our data
# Transform - Transform method is transforming all the features using the respective mean and variance,

# We want our test data to be a completely new and a surprise set for our model
print("\n5. Feature Scaling:")
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
print(f'X_train: \n {X_train}')
X_test = sc.transform(X_test)
print(f'X_test: \n {X_test}')

"""## Finding the Best K-Value """
print("\n6. Finding the Best K-Value: ")

error = []
# Select K value from the Minimum Error rate
for i in range(1, 40):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, Y_train)
    predicted_Y_test = model.predict(X_test)
    error.append(np.mean(predicted_Y_test != Y_test))
print(f'Mean Error: {min(error)}')

# Plot the Graph between the Error Rate Vs K Value
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate Vs K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

"""## Training """
print("\n7. Training:  trained")

model = KNeighborsClassifier(n_neighbors= 15, metric='minkowski', p=2)
model.fit(X_train, Y_train)

# Prediction for all Test Data
print("\n8. Prediction for all Test Data:")
Y_test_predicted = model.predict(X_test)

# Display Result gave by the Model and Actual Result
print(np.concatenate((Y_test_predicted.reshape(len(Y_test_predicted), 1), Y_test.reshape(len(Y_test), 1)), 1))

"""### Evaluating Model - CONFUSION MATRIX """
print("\n9. Evaluating Model - CONFUSION MATRIX: ")
cm = confusion_matrix(Y_test, Y_test_predicted)
print(f' Confusion Matrix: \n {cm}')
print(" Accuracy of the Model: {0}%".format(accuracy_score(Y_test, Y_test_predicted) * 100))

"""## Predicting, whether new customer with Age & Salary will Buy or Not"""
print("\n\n10. Salary Estimator: ")
age = int(input("Enter New Customer Age: "))
salary = int(input("Enter New Customer Salary: "))
newCust = [[age, salary]]
result = model.predict(sc.transform(newCust))
print(result)

if result == 1:
    print("Customer will Buy")
else:
    print("Customer won't Buy")
