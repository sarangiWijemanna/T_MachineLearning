# Day 4 - _Titanic Survival Prediction_NAIVEBAYES

""" ## Importing Libraries"""
import pandas as pd
import numpy as np

""" ## Load and Summarized Dataset"""
print("1. Load and Summarized Dataset: ")
# Choose File from the Local Machine when Using Google Colab
#   from google.colab import files
#   uploaded = files.upload()

# Load Dataset
dataset = pd.read_csv("Titanic Survival Prediction Dataset.csv")

# Summarised Dataset
print(f' * Shape of dataset: {dataset.shape}')
print(f' * Context of dataset: \n {dataset.head(5)}')

""" ## Mapping Text Data to Binary Data"""
print("\n2. Mapping Text Data to Binary Data: ")

Sex_set = set(dataset['Sex'])
print(f' * Set of Sex: {Sex_set}')

dataset['Sex'] = dataset['Sex'].map({'male': 1, 'female': 0}).astype(int)
print(f' * Dataset after mapped: \n{dataset}')

""" ## Segregate Dataset into X and Y"""
print("\n3. Segregate Dataset into X and Y:")
# Input (X) = pClass, Age, Sex, Fare
X = dataset.drop('Survived', axis='columns')
print(f' * X  = \n{X}')

# Output (Y) = Survived (1) or Not (0)
Y = dataset.Survived
print(f'\n * Y  = \n{Y}')

""" ## Finding & Removing NaN Values from the X Features"""
print("\n4. Finding & Removing NaN Values from the X Features: ")

# Find where NaN values are existed in the X features
print(f' * NaN existed in: {X.columns[X.isna().any()]}')

# Removed with those NaN vales with the Mean of relevant Feature
X.Age = X.Age.fillna(X.Age.mean())

# Test again to check any NaN value
print(f' * NaN existed or Not: {X.columns[X.isna().any()]}')

""" ## Splitting Dataset into Train (75%) and Test(25%)"""
print("\n5. Splitting Dataset into Train & Test:")
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of X_test: {X_test.shape}')
print(f'Shape of y_train: {Y_train.shape}')
print(f'Shape of y_test: {Y_test.shape}')

""" ## Selected Algorithm & Training"""
print("\n6. Selected Algorithm & Training: GaussianNB ")
# Used = Naives Bayes
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, Y_train)

""" Evaluation"""
print("\n7. Evaluation: ")

# Prediction for all Test Data
print(" * Prediction for all Test Data: ")
Y_predicted = model.predict(X_test)
print(np.column_stack((Y_predicted, Y_test)))

# Accuracy of our Model
print(" * \nAccuracy of our Model:")
from sklearn.metrics import accuracy_score

print("Accuracy of the Model: {0}%".format(accuracy_score(Y_test, Y_predicted) * 100))

""" ## Prediction """
print("\n\n8. Predicting, whether Person Survived or Not: ")
# Predicting, whether Person Survived or Not
Pclass = int(input("Enter Person's class number: "))
Sex = int(input("Enter Person's Gender 0-female 1-male(0 or 1): "))
age = int(input("Enter Person's Age: "))
Fare = float(input("Enter Person's Fare: "))
person = [[Pclass, Sex, age, Fare]]

result = model.predict(person)
print(f'Result: {result}')

if result == 1:
    print("Sure, Person will be Survived..!")
else:
    print("Bad, Person will not be Survived..!")

"""## Try with the Different Method """
print("\n\n8. Try with the Different Method: ")
# Used = Naives Bayes
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB

model1 = GaussianNB()
model2 = MultinomialNB()
model3 = ComplementNB()
model4 = BernoulliNB()
model5 = CategoricalNB()

model1.fit(X_train, Y_train)
model2.fit(X_train, Y_train)
model3.fit(X_train, Y_train)
model4.fit(X_train, Y_train)
model5.fit(X_train, Y_train)

y_prediction_Model1 = model1.predict(X_test)
y_prediction_Model2 = model2.predict(X_test)
y_prediction_Model3 = model3.predict(X_test)
y_prediction_Model4 = model4.predict(X_test)
y_prediction_Model5 = model5.predict(X_test)

print("Accuracy of the Model 1: {0}%".format(accuracy_score(Y_test, y_prediction_Model1) * 100))
print("Accuracy of the Model 2: {0}%".format(accuracy_score(Y_test, y_prediction_Model2) * 100))
print("Accuracy of the Model 3: {0}%".format(accuracy_score(Y_test, y_prediction_Model3) * 100))
print("Accuracy of the Model 4: {0}%".format(accuracy_score(Y_test, y_prediction_Model4) * 100))
print("Accuracy of the Model 5: {0}%".format(accuracy_score(Y_test, y_prediction_Model5) * 100))
