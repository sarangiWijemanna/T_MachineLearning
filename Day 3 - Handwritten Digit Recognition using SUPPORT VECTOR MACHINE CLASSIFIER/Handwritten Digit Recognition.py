# Day 3 - Handwritten Digit Recognition using SVM Classifier


import numpy as np
from sklearn.datasets import load_digits

print("Day 3 - Handwritten Digit Recognition using SVM Classifier")

"""## Load Dataset & Summarize Dataset"""
print("\n1. Load Dataset & Summarize Dataset: ")

# Load Dataset
dataset = load_digits()

# Summarize Dataset
print(f'Data: \n{dataset.data}')
print(f'Target: \n{dataset.target}')
print(f'Images: \n{dataset.images}')

print(f'Data Shape: {dataset.data.shape}')
print(f'Image Shape: {dataset.images.shape}')

dataImageLength = len(dataset.images)
print(f'Image Length: {dataImageLength}')

"""## Visualize the Dataset """
print("\n2. Visualize the Dataset: ")
n = int(input("Sample Number out of Total Samples 1797: "))  # Sample Number out of Total Samples 1797
import matplotlib.pyplot as plt

plt.gray()
plt.matshow(dataset.images[n])
print(f'Image[{n}]: \n{dataset.images[n]}')
plt.show()

"""## Segregate Dataset into X(Input/IndependentVariable) & Y(Output/DependentVariable) """
# Input - Pixel | Output - Class
print("\n3. Segregate Dataset into X and Y: ")

X = dataset.images.reshape((dataImageLength, -1))
print(f'X(input): \n{X}')
Y = dataset.target
print(f'Y(output): {Y}')

"""## Splitting Dataset into Train & Test """
print("\n4. Splitting Dataset into Train & Test: ")
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of X_test: {X_test.shape}')
print(f'Shape of Y_train: {Y_train.shape}')
print(f'Shape of Y_test: {Y_test.shape}')

"""## Training """
print("\n5. Training: trained..!")
from sklearn import svm

# Support Vector Classifier
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

"""## Evaluate Model """
print("\n6. Evaluate Model: ")
# Prediction for Test Data
Y_Prediction = model.predict(X_test)
print(
    f'Comparison Y_Prediction vs Y_test: \n{np.concatenate((Y_Prediction.reshape(len(Y_Prediction), 1), Y_test.reshape(len(Y_test), 1)), 1)}')

# Evaluate Model - Accuracy Score
from sklearn.metrics import accuracy_score

print("Accuracy of the Model: {0}%".format(accuracy_score(Y_test, Y_Prediction) * 100))

"""## Predicting, what the digit is from Test Data """
print("\n7. Predicting, what the digit is from Test Data: ")

n = int(input("Sample Number out of Total Samples 1797: "))
result = model.predict(dataset.images[n].reshape((1, -1)))
plt.imshow(dataset.images[n], cmap=plt.cm.gray_r, interpolation='nearest')
print(result)
print("\n")
plt.axis('off')
plt.title("Result: "'%i' % result)
plt.show()

"""## Try with the Different Method """
print("8. Try with the Different Method: ")
from sklearn import svm

# Default gamma = 0.1, Degree = 3, C = 1
model1 = svm.SVC(kernel='linear')
model2 = svm.SVC(kernel='rbf')
model3 = svm.SVC(gamma=0.001)
model4 = svm.SVC(gamma=0.001, C=0.1)
model5 = svm.SVC(gamma=0.001, C=0.4)

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
