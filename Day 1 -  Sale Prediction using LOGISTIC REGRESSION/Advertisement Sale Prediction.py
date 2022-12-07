"# Day 1 - Advertisement  Sale Prediction from Existing Customer - Logistic Regression"""

"""### *Importing Libraries*"""
import pandas as pd  # Perform Loading csv Dataset
import numpy as np  # Perform Array Operations

"""### *Choose Dataset file from Local Directory*"""
# If using Colab, You can use this command to Choose the Dataset
# from google.colab import files
# uploaded = files.upload()

"""### *Load Dataset*"""
dataset = pd.read_csv('Advertisement Sale Dataset.csv')

"""### *Summarize Dataset*"""
print(dataset.shape)          # Gives number of Rows and Columns in dataset
print(dataset.head(5))        # Gives first 5 Readings from the dataset


"""### *Segregate Dataset into X(Input/IndependentVariable) & Y(Output/DependentVariable)*"""
X = dataset.iloc[0:, 0:-1].values  # Select Inputs Columns with all Rows
Y = dataset.iloc[0:, -1].values  # Select Outputs Columns with all Rows

"""### *Splitting Dataset into Train & Test*"""
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

"""### *Feature Scaling*
#### To perform all features contribute equally to the final result
#### Fit_Transform - Calculating the mean and variance of each of the features present in our dataset
#### Transform - Transform method is transforming all the features using the respective mean and variance, 
#### We want our test data to be a completely new and a surprise set for our model and not same as training dataset
"""
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""### *Training*"""
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=0)
model.fit(X_train, Y_train)

"""### *Prediction for all Test Data*"""

y_pred = model.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), Y_test.reshape(len(Y_test), 1)), 1))


"""### *Evaluating Model - CONFUSION MATRIX*"""
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(Y_test, y_pred)
print("\n\nConfusion Matrix: ")
print(cm)
print("Accuracy of the Model: {0}%".format(accuracy_score(Y_test, y_pred) * 100))


"""### *Predicting, whether new customer with Age & Salary will Buy or Not*"""
age = int(input("Enter New Customer Age: "))
salary = int(input("Enter New Customer Salary: "))
newCust = [[age, salary]]

result = model.predict(sc.transform(newCust))
# print(result)

if result == 1:
    print("Customer will Buy")
else:
    print("Customer won't Buy")


