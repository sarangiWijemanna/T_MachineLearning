"# Day 8 | Breast Cancer Detection using various ML Algorithm - Evaluation"""

"""### *Importing Libraries*"""
import pandas as pd  # Perform Loading csv Dataset
import numpy as np  # Perform Array Operations
from matplotlib import pyplot as plt  # Perform Plotting

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
print(dataset)

"""## Segregate Dataset into X(Input/IndependentVariable) & Y(Output/DependentVariable)"""
print("\n3. Segregate Dataset into X and Y:")
X = dataset.iloc[0:, 2:32]  # Select Inputs Columns with all Rows
print(f'X = \n{X}')
Y = dataset.iloc[0:, 1].values  # Select Outputs Columns with all Rows
print(f'Y = \n{Y}')

"""## Splitting Dataset into Train & Test """
print("\n4. Splitting Dataset into Train & Test:")
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
print(f'    Shape of X_train: {X_train.shape}')
print(f'    Shape of X_test: {X_test.shape}')
print(f'    Shape of y_train: {Y_train.shape}')
print(f'    Shape of y_test: {Y_test.shape}')

"""## Feature Scaling*
#### To perform all features contribute equally to the final result
#### Fit_Transform - Calculating the mean and variance of each of the features present in our dataset
#### Transform - Transform method is transforming all the features using the respective mean and variance, 
#### We want our test data to be a completely new and a surprise set for our model and not same as training dataset
"""
print("\n5. Feature Scaling:")
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(f'    X_train: \n{X_train}')
print(f'\n    X_test: \n{X_test}')

"""## Validating some ML algorithm by its accuracy - Model Score"""
print("\n6. Validating some ML algorithm by its accuracy - Model Score:")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

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
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), Y_test.reshape(len(Y_test), 1)), 1))




