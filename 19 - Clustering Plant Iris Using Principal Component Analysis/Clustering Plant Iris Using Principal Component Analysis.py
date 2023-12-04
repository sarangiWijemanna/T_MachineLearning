" Day 19 | Clustering Plant Iris Using Principal Component Analysis - UNSUPERVISED LEARNING"""
print("Day 19 | Clustering Plant Iris Using Principal Component Analysis - UNSUPERVISED LEARNING")

" Importing the basic libraries"""

from sklearn import datasets
import matplotlib.pyplot as plt

" Collecting Dataset """

# Importing the dataset"""
dataset = datasets.load_iris()

" Dataset Segregation"""
X = dataset.data
print("\nX= \n", X)

Y = dataset.target
print("\nY= \n", Y)

names = dataset.target_names
print("\nnames= \n", names)

"Fitting the PCA clustering to the dataset with n=2"""
from sklearn.decomposition import PCA

model = PCA(n_components=2)  # Number of components to keep
y_means = model.fit(X).transform(X)
print("\ny_means= \n", y_means)

"Variance Percentage"""
plt.figure()
colors = ['red', 'green', 'orange']

for color, i, target_name in zip(colors, [0, 1, 2], names):
    plt.scatter(y_means[Y == i, 0],
                y_means[Y == i, 1],
                color=color,
                lw=2,
                label=target_name)
plt.title('IRIS Clustering')
plt.savefig('Clustering Plant Iris Using Principal Component Analysis.png')
plt.show()
