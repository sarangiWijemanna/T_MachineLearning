"# Day 8 | Tuning Parameter - Breast Cancer Detection using various ML Algorithm - Evaluation"""

"""### *Importing Libraries*"""
import pandas as pd  # Perform Loading csv Dataset
import numpy as np  # Perform Array Operations

from matplotlib import pyplot as plt  # Perform Plotting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

""" ## Load and Summarized Dataset"""
print("1. Load and Summarized Dataset:")
# Choose Dataset file from Local Directory
# If using Colab, You can use this command to Choose the Dataset
# from google.colab import files
# uploaded = files.upload()

# Load Dataset
dataset = pd.read_csv('Breast Cancer Detection Datasheet.csv')

# Summarize Dataset
print(dataset.shape)  # Gives number of Rows and Columns in dataset
print(dataset.head(5))  # Gives first 5 Readings from the dataset

"""## Mapping Class String Values to Numbers """
print("\n2. Mapping Class String Values to Numbers:")
dataset['diagnosis'] = dataset['diagnosis'].map({'B': 0, 'M': 1}).astype(int)

"""## Segregate Dataset into X(Input/IndependentVariable) & Y(Output/DependentVariable)"""
print("\n3. Segregate Dataset into X and Y:")
X = dataset.iloc[0:, 2:32]  # Select Inputs Columns with all Rows
print(f'X = \n{X}')
Y = dataset.iloc[0:, 1].values  # Select Outputs Columns with all Rows
print(f'Y = \n{Y}')

"""## Splitting Dataset into Train & Test """
print("\n4. Splitting Dataset into Train & Test:")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

"""## Feature Scaling """
print("\n5. Feature Scaling:")
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""## ML algorithm"""
print("\n6. ML algorithm :")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

"""## Task: Find Tuning Parameters"""
print("\n7. Find Tuning Parameters :")
# 1. Finding Best K-vale for KNN
'''error = []
#   Select K value from the Minimum Error rate
for i in range(1, 40):
    model1 = KNeighborsClassifier(n_neighbors=i)
    model1.fit(X_train, Y_train)
    predicted_Y_test = model1.predict(X_test)
    error.append(np.mean(predicted_Y_test != Y_test))
print(f'Mean Error: {min(error)}')  # @ K =15

#   Plot the Graph between the Error Rate Vs K Value
# plt.figure(figsize=(12, 6))
# plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',markerfacecolor='blue', markersize=10)
# plt.title('Error Rate Vs K Value')
# plt.xlabel('K Value')
# plt.ylabel('Mean Error')
#  plt.grid(True)
# plt.show()'''
#    Parameters = n_neighbors=15, metric='minkowski', p=2

# 2. SVM Classifier Tuning Parameter
# Parameters = kernel='rbf',  Default gamma = 0.1, Degree = 3, C = 1

# DECISION TREE Tuning Parameter
#   Finding best max_depth Value
'''accuracy = []
from sklearn.metrics import accuracy_score

for j in range(1, 10):
    model3 = DecisionTreeClassifier(max_depth=j, random_state=0)
    model3.fit(X_train, Y_train)
    Y_prediction = model3.predict(X_test)
    score = accuracy_score(Y_test, Y_prediction)
    accuracy.append(score)

# plt.figure(figsize=(12, 6))
# plt.plot(range(1, 10), accuracy, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
# plt.title('Finding best Max_Depth')
# plt.xlabel('Max_Depth Prediction')
# plt.ylabel('score')'''
# Parameters = criterion='entropy', max_depth=2, random_state=0

#  RANDOM FOREST
# Parameters = n_estimators=500, criterion='gini', max_depth=25,  bootstrap=False, random_state=0

"""## Validating some ML algorithm by its accuracy - Model Score"""
print("\n8. Validating some ML algorithm by its accuracy - Model Score:")
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='auto', random_state=0)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(n_neighbors=15, metric='minkowski', p=2)))
models.append(('CART', DecisionTreeClassifier(criterion='entropy', max_depth=26)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(kernel='rbf')))
models.append(
    ('RF', RandomForestClassifier(n_estimators=22, criterion='entropy', bootstrap=False, random_state=0)))

results = []
names = []
res = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=None)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    res.append(cv_results.mean())
    print('%s: %f' % (name, cv_results.mean()))

plt.ylim(.900, .999)
plt.bar(names, res, color='maroon', width=0.6)

plt.title('Algorithm Comparison')
plt.show()

"""## Training & Prediction using the algorithm with high accuracy """
print("\n7. Training & Prediction using the algorithm with high accuracy:")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

model = LogisticRegression(solver='liblinear', multi_class='auto', random_state=0)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), Y_test.reshape(len(Y_test), 1)), 1))

