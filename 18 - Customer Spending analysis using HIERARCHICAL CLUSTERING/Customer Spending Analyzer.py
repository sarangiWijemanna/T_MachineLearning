" Day 18 | Customer Spending Analysis Using Hierarchical Clustering - UNSUPERVISED LEARNING"""
print("Day 18 | Customer Spending Analysis Using Hierarchical Clustering - UNSUPERVISED LEARNING")

" Importing the basic libraries"""

import pandas as pd
import matplotlib.pyplot as plt

"Load Dataset from Local Directory"""

# Importing the dataset"""
dataset = pd.read_csv('Customer Spending Analysis Dataset.csv')

# Summarize Dataset
print("\nShape of Dataset = ", dataset.shape)
print("\nDescribe of Dataset = \n", dataset.describe())
print("\nHead of Dataset = \n", dataset.head(5))

"Label Encoding"""
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
dataset['Gender'] = label_encoder.fit_transform(dataset['Gender'])
print("\nHead of Dataset After Encoding = \n", dataset.head())

"Dendrogram Data visualization"""
import scipy.cluster.hierarchy as clus

plt.figure(1, figsize=(16, 8))
dendrogram = clus.dendrogram(clus.linkage(dataset, method="ward"))

plt.title('Dendrogram Tree Graph')
plt.xlabel('Customers')
plt.ylabel('Distances')
plt.savefig('Dendrogram Tree Graph.png')
plt.show()

" Fitting the Hierarchical clustering to the dataset with n=5"""
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='average')
y_means = model.fit_predict(dataset)
print("\ny_means = \n", y_means)

""" Visualizing the number of clusters n=5

Cluster 1: Customers with Medium Income and Medium Spending

Cluster 2: Customers with High Income and High Spending

Cluster 3: Customers with Low Income and Low Spending

Cluster 4: Customers with High Income and Low Spending

Cluster 5: Customers with Low Income and High Spending
"""

X = dataset.iloc[:, [3, 4]].values
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s=50, c='purple', label='Cluster 1')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s=50, c='orange', label='Cluster 2')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s=50, c='red', label='Cluster 3')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s=50, c='green', label='Cluster 4')
plt.scatter(X[y_means == 4, 0], X[y_means == 4, 1], s=50, c='blue', label='Cluster 5')
plt.title('Customer Income Spent Analysis - Hierarchical Clustering')
plt.xlabel('Income')
plt.ylabel('Spent')
plt.savefig('Customer Income Spent Analysis - HC.png')
plt.show()
