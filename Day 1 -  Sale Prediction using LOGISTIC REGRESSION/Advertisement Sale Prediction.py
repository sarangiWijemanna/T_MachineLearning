# Day 3 Advertisement  Sale Prediction from Existing Customer - Logistic Regression


"""### *Importing Libraries*"""
import pandas as pd  # Perform Loading csv Dataset
import numpy as np  # Perform Array Operations

"""### *Choose Dataset file from Local Directory*"""
# If using Colab, You can use this command to Choose the Dataset
# from google.colab import files
# uploaded = files.upload()

"""### *Load Dataset*"""
dataset = pd.read_csv('Advertisement Sale Dataset.csv')

"""### *Summarize Dataset*"""
print(dataset.shape)    # Gives number of Rows and Columns in dataset
print(dataset.head(5))  # Gives first 5 Readings from the dataset



