" Day 17 | Income Spent Analysis Using K-Means Clustering  - UNSUPERVISED LEARNING""""

"Importing the basic libraries"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""## Importing the dataset"""
dataset = pd.read_csv('Income Spent Dataset.csv')

"""### Summarize Dataset"""
print(dataset.shape)
print(dataset.describe())
print(dataset.head(5))

"""### Segregate & Zipping Dataset"""
Income = dataset['INCOME'].values
Spend = dataset['SPEND'].values
X = np.array(list(zip(Income, Spend)))

"""### Finding the Optimized K Value"""
from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(X)
    wcss.append(km.inertia_)
plt.plot(range(1, 11), wcss, color="red", marker="8")
plt.title('Optimal K Value')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('Finding the Optimized K Value.png')
plt.show()

"""### Fitting the k-means to the dataset with k=4"""

model = KMeans(n_clusters=4, random_state=0)
y_means = model.fit_predict(X)

"""### Visualizing the clusters for k=4

Cluster 1: Customers with medium income and low spend

Cluster 2: Customers with high income and medium to high spend

Cluster 3: Customers with low income

Cluster 4: Customers with medium income but high spend
"""

plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s=50, c='brown', label='1')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s=50, c='blue', label='2')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s=50, c='green', label='3')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s=50, c='cyan', label='4')
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=100, marker='s', c='red', label='Centroids')
plt.title('Income Spent Analysis')
plt.xlabel('Income')
plt.ylabel('Spent')
plt.legend()
plt.savefig('Income Spent Analysis - K Means Clustering.png')
plt.show()
