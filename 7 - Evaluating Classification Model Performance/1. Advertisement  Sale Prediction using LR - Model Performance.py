"# Day 7 - Model Performance - Advertisement  Sale Prediction from Existing Customer - Logistic Regression"""

"""### *Importing Libraries*"""
import pandas as pd  # Perform Loading csv Dataset
import numpy as np  # Perform Array Operations

""" ## Load and Summarized Dataset"""
print("1. Load and Summarized Dataset:")
# Choose Dataset file from Local Directory
# If using Colab, You can use this command to Choose the Dataset
# from google.colab import files
# uploaded = files.upload()

# Load Dataset
dataset = pd.read_csv('Advertisement Sale Dataset.csv')

# Summarize Dataset
print(dataset.shape)  # Gives number of Rows and Columns in dataset
print(dataset.head(5))  # Gives first 5 Readings from the dataset

"""## Segregate Dataset into X(Input/IndependentVariable) & Y(Output/DependentVariable)"""
print("2. Segregate Dataset into X and Y:")
X = dataset.iloc[0:, 0:-1].values  # Select Inputs Columns with all Rows
Y = dataset.iloc[0:, -1].values  # Select Outputs Columns with all Rows

"""## Splitting Dataset into Train & Test """
print("3. Splitting Dataset into Train & Test:")
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

"""## Feature Scaling*
#### To perform all features contribute equally to the final result
#### Fit_Transform - Calculating the mean and variance of each of the features present in our dataset
#### Transform - Transform method is transforming all the features using the respective mean and variance, 
#### We want our test data to be a completely new and a surprise set for our model and not same as training dataset
"""
print("4. Feature Scaling:")
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""## Training """
print("5. Training: Using Logistic Regression")
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=0)
model.fit(X_train, Y_train)

"""## Prediction for all Test Data """
print("6. Prediction for all Test Data:")
Y_prediction = model.predict(X_test)
print(np.concatenate((Y_prediction.reshape(len(Y_prediction), 1), Y_test.reshape(len(Y_test), 1)), 1))

""" ## *Advertisement Sale Estimation | Logistic Regression*"""
print("\n\n***Advertisement Sale Estimation | Logistic Regression***")
"""## Evaluating Model """
print("7. Evaluating Model:")

# 1. CONFUSION MATRIX
print(" 7.1. CONFUSION MATRIX: ")
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(Y_test, Y_prediction)
print(f' {cm}')

# 2. Accuracy_Score
print(" 7.2. Accuracy Score:")
from sklearn.metrics import accuracy_score

print("      Accuracy of the Model: {0}%".format(accuracy_score(Y_test, Y_prediction) * 100))

"""## Receiver Operating Curve - ROC Curve """
print(" 7.3. Receiver Operating Curve - ROC Curve:")
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

nsProbability = [0 for _ in range(len(Y_test))]
lsProbability = model.predict_proba(X_test)
# keep probabilities for the positive outcome only
lsProbability = lsProbability[:, 1]
# calculate scores
nsAUC = roc_auc_score(Y_test, nsProbability)
lrAUC = roc_auc_score(Y_test, lsProbability)
# summarize scores
print('   No Skill: ROC AUC=%.3f' % (nsAUC * 100))
print('   Logistic Skill: ROC AUC=%.3f' % (lrAUC * 100))
# calculate roc curves
nsFP, nsTP, _ = roc_curve(Y_test, nsProbability)
lrFP, lrTP, _ = roc_curve(Y_test, lsProbability)
# plot the roc curve for the model
plt.plot(nsFP, nsTP, linestyle='--', label='No Skill')
plt.plot(lrFP, lrTP, marker='*', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
plt.show()

"""##Cross Validation Score"""
print(" 7.4. Cross Validation Score:")
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

kfold = KFold(n_splits=10, random_state=100, shuffle=True)
result = cross_val_score(model, X, Y, cv=kfold)
print("  CROSS VALIDATION SCORE: %.2f%%" % (result.mean() * 100.0))

"""## Stratified K-fold Cross Validation"""
print(" 7.5. Stratified K-fold Cross Validation:")
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

skfold = StratifiedKFold(n_splits=3, random_state=100, shuffle=True)
model_skfold = LogisticRegression()
results_skfold = cross_val_score(model_skfold, X, Y, cv=skfold)
print("   STRATIFIED K-FOLD Score: %.2f%%" % (results_skfold.mean() * 100.0))

"""## Cumulative Accuracy Profile (CAP) Curve"""
print(" 7.5. Cumulative Accuracy Profile (CAP) Curve:")
total = len(Y_test)
print(f'   Total observations: {total}')
class_1_count = np.sum(Y_test)
print(f'   class_1_count: {class_1_count}')
class_0_count = total - class_1_count
plt.plot([0, total], [0, class_1_count], c='r', linestyle='--', label='Random Model')

plt.plot([0, class_1_count, total],
         [0, class_1_count, class_1_count],
         c='grey',
         linewidth=2,
         label='Perfect Model')

probs = model.predict_proba(X_test)
probs = probs[:, 1]
model_y = [y for _, y in sorted(zip(probs, Y_test), reverse=True)]
y_values = np.append([0], np.cumsum(model_y))
x_values = np.arange(0, total + 1)

plt.plot(x_values,
         y_values,
         c='b',
         label='LR Classifier',
         linewidth=4)

index = int((50 * total / 100))

# 50% Vertical line from x-axis
plt.plot([index, index], [0, y_values[index]], c='g', linestyle='--')

# Horizontal line to y-axis from prediction model
plt.plot([0, index], [y_values[index], y_values[index]], c='g', linestyle='--')

class_1_observed = y_values[index] * 100 / max(y_values)
plt.xlabel('Total observations')
plt.ylabel('Class 1 observations')
plt.title('Cumulative Accuracy Profile')
plt.legend(loc='lower right')

""" ## *Advertisement Sale Estimation | K-NEAREST NEIGHBOUR model*"""
print("\n\n***Advertisement Sale Estimation | K-NEAREST NEIGHBOUR model***")
"""## Finding the Best K-Value """

error = []
from sklearn.neighbors import KNeighborsClassifier  # Find K Value and Training Model

# Select K value from the Minimum Error rate
for i in range(1, 40):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, Y_train)
    predicted_Y_test = model.predict(X_test)
    error.append(np.mean(predicted_Y_test != Y_test))
# print(f'Mean Error: {min(error)}')

# Plot the Graph between the Error Rate Vs K Value
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate Vs K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

"""## Training """
from sklearn.neighbors import KNeighborsClassifier  # Find K Value and Training Model

model = KNeighborsClassifier(n_neighbors=15, metric='minkowski', p=2)
model.fit(X_train, Y_train)

# Prediction for all Test Data
Y_test_predicted = model.predict(X_test)

# Display Result gave by the Model and Actual Result
# print(np.concatenate((Y_test_predicted.reshape(len(Y_test_predicted), 1), Y_test.reshape(len(Y_test), 1)), 1))

"""## Evaluating Model """
print("7. Evaluating Model:")

# 1. CONFUSION MATRIX
print(" 7.1. CONFUSION MATRIX: ")
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(Y_test, Y_prediction)
print(f'{cm}')

# 2. Accuracy_Score
print(" 7.2. Accuracy Score:")
from sklearn.metrics import accuracy_score

print("   Accuracy of the Model: {0}%".format(accuracy_score(Y_test, Y_prediction) * 100))

"""## Receiver Operating Curve - ROC Curve """
print(" 7.3. Receiver Operating Curve - ROC Curve:")
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

nsProbability = [0 for _ in range(len(Y_test))]
lsProbability = model.predict_proba(X_test)
# keep probabilities for the positive outcome only
lsProbability = lsProbability[:, 1]
# calculate scores
nsAUC = roc_auc_score(Y_test, nsProbability)
lrAUC = roc_auc_score(Y_test, lsProbability)
# summarize scores
print('   No Skill: ROC AUC=%.3f' % (nsAUC * 100))
print('   KNN Skill: ROC AUC=%.3f' % (lrAUC * 100))
# calculate roc curves
nsFP, nsTP, _ = roc_curve(Y_test, nsProbability)
lrFP, lrTP, _ = roc_curve(Y_test, lsProbability)
# plot the roc curve for the model
plt.plot(nsFP, nsTP, linestyle='--', label='No Skill')
plt.plot(lrFP, lrTP, marker='*', label='KNN')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
plt.show()

"""## Cross Validation Score"""
print(" 7.4. Cross Validation Score:")
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

kfold = KFold(n_splits=10, random_state=100, shuffle=True)
result = cross_val_score(model, X, Y, cv=kfold)
print("   Cross Validation Score : %.2f%%" % (result.mean() * 100.0))

"""## Stratified K-fold Cross Validation"""
print(" 7.5. Stratified K-fold Cross Validation:")
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

skfold = StratifiedKFold(n_splits=3, random_state=100, shuffle=True)
model_skfold = LogisticRegression()
results_skfold = cross_val_score(model_skfold, X, Y, cv=skfold)
print("   STRATIFIED K-FOLD Score: %.2f%%" % (results_skfold.mean() * 100.0))

"""## Cumulative Accuracy Profile (CAP) Curve"""
print(" 7.5. Cumulative Accuracy Profile (CAP) Curve:")
total = len(Y_test)
print(f'   Total Observations: {total}')
class_1_count = np.sum(Y_test)
print(f'   class_1_count: {class_1_count}')
class_0_count = total - class_1_count
plt.plot([0, total], [0, class_1_count], c='r', linestyle='--', label='Random Model')

plt.plot([0, class_1_count, total],
         [0, class_1_count, class_1_count],
         c='grey',
         linewidth=2,
         label='Perfect Model')

probs = model.predict_proba(X_test)
probs = probs[:, 1]
model_y = [y for _, y in sorted(zip(probs, Y_test), reverse=True)]
y_values = np.append([0], np.cumsum(model_y))
x_values = np.arange(0, total + 1)

plt.plot(x_values,
         y_values,
         c='b',
         label='KNN Classifier',
         linewidth=4)

index = int((50 * total / 100))

# 50% Vertical line from x-axis
plt.plot([index, index], [0, y_values[index]], c='g', linestyle='--')

# Horizontal line to y-axis from prediction model
plt.plot([0, index], [y_values[index], y_values[index]], c='g', linestyle='--')

class_1_observed = y_values[index] * 100 / max(y_values)
plt.xlabel('Total observations')
plt.ylabel('Class 1 observations')
plt.title('Cumulative Accuracy Profile')
plt.legend(loc='lower right')
plt.show()
