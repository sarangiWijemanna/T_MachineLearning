# 💛 UNSUPERVISED LEARNING | Hierarchical Clustering ⚖🌲

## Day 18 | Customer Spending Analysis 🙂

### Step 1 :  Find Problem 

- To categorize the information based on the amount spent by customers.

### Step 2 : Collect Dataset 

- Input : Amount Spent analysis and segregate data as different groups.

### Step 3 : Load and Summered Dataset

- Import csv file from local directory.
- Summarize data.
  - Shape
  - Describe
  
### Step 4: Label Encoding

- We need to convert string/text data to numerical values.
  - Here, we have text data in the Gender column.
    - Method 1:  We can use the ``map`` function to convert text data to numerical data.
    - Method 2: We can use ``sklearn.preprocessing`` to encode text data.

### Step 4 : Algorithm | Hierarchical Clustering 

> Definition :

- Group or cluster the data in Hierarchical way based on their similarity or dissimilarity.
- It is an unsupervised learning method (it doesn't require any prior knowledge about the data).
- In hierarchical clustering, the data points are iteratively grouped together into clusters, 
  - where the similarity between the data points within the same cluster is high and the similarity between the data points in different clusters is low. 
  - The process starts by considering each data point as a separate cluster and then merging them into larger clusters based on their similarity. 

> Method :

- Compute a distance matrix.
  - Euclidean Distance 
    - Compute Distance between two clusters by drawing straight line between them.
- Need to determine which distance is needed to take (Linkage Criteria).
  - Single Linkage 
  - Complete Linkage
  - Memory Average Linkage
  - Ward (Default) Linkage

> Types :

- Agglomerative Hierarchical Clustering:

  - This method starts with individual data points and progressively merges them into clusters until all data points are in a single cluster. 
  - At the beginning of the process, each data point is treated as a separate cluster, 
  - and then pairs of clusters are merged based on their similarity. 
  - This process continues until all data points are in a single cluster.

- Divisive Hierarchical Clustering:

  - This method starts with all the data points in a single cluster and then divides it into smaller clusters until each data point is in its own cluster. 
  - This method begins with a single large cluster and progressively splits it into smaller clusters based on the dissimilarity between the data points.


## Step 5 : Data Visualization

> Screenshot for Output of Hierarchical Clustering : 

<img align="center" src="Dendrogram Tree Graph.png" alt="icon"/>


## Step 6 : Fitting Model to Hierarchical Clustering Algorithm

- We need to fitting the HC to the dataset with n = 5.

  - Visualizing the number of clusters n=5

    - Cluster 1: Customers with Medium Income and Medium Spending

    - Cluster 2: Customers with High Income and High Spending

    - Cluster 3: Customers with Low Income and Low Spending

    - Cluster 4: Customers with High Income and Low Spending

    - Cluster 5: Customers with Low Income and High Spending

> Screenshot for Output of Hierarchical Clustering : 

<img align="center" src="Customer Income Spent Analysis - HC.png" alt="icon"/>
          
