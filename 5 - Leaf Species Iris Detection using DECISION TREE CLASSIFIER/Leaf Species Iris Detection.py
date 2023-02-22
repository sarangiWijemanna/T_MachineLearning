# Day-5 | Leaf Species Detection | DECISION TREE

# Import basic Libraries
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

""" ## Load and Summarized Dataset"""
print("1. Load and Summarized Dataset: ")

# Load Dataset
dataset = load_iris()

# Summarized Dataset
print(dataset.data)
print(dataset.target)
print(dataset.data.shape)

""" ## Segregate Dataset into X and Y"""
print("2. Segregate Dataset into X and Y: ")
# X = (Input/IndependentVariable)
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
print(X)

# Y = (Output/DependentVariable)*"""
Y = dataset.target
print(Y)

""" ## Splitting Dataset into Train (75%) and Test(25%)"""
print("\n3. Splitting Dataset into Train & Test:")
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
print(X_train.shape)

"""## Finding best max_depth Value"""
print("\n4. Finding best max_depth Value: ")

accuracy = []
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

for i in range(1, 10):
    model = DecisionTreeClassifier(max_depth=i, random_state=0)
    model.fit(X_train, Y_train)
    Y_prediction = model.predict(X_test)
    score = accuracy_score(Y_test, Y_prediction)
    accuracy.append(score)

plt.figure(figsize=(12, 6))
plt.plot(range(1, 10), accuracy, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Finding best Max_Depth')
plt.xlabel('Max_Depth Prediction')
plt.ylabel('score')
plt.show()

"""## Training """
print("\n5. Training: ")
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
model.fit(X_train, Y_train)

"""## Prediction """
print("\n6. Prediction: ")
Y_prediction = model.predict(X_test)
print(np.concatenate((Y_prediction.reshape(len(Y_prediction), 1), Y_test.reshape(len(Y_test), 1)), 1))

"""### *Accuracy Score*"""
from sklearn.metrics import accuracy_score
print("Accuracy of the Model: {0}%".format(accuracy_score(Y_test, Y_prediction) * 100))
