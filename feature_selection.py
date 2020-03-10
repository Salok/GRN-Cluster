import pandas as pd
import numpy as np
import scipy as sp
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

"""Reduce the set of features by trying to find a reduced subset with the same amount of information as the whole set"""

dataset = pd.read_csv("/home/cig/Desktop/PHC/full_feature_extraction_dataset.csv")
hubs_score = pd.read_csv("/home/cig/Desktop/PHC/hubs_scores.csv")

X = dataset.values

"""Versión cutre de pruebas, seleccionamos por varianza y luego tiramos un kmeans"""
selector = VarianceThreshold(threshold=4).fit(X)
selected_variables = selector.get_support()

X_filtered = selector.transform(X)

kmeans = KMeans(n_clusters=10, verbose=True).fit(X_filtered)

predictions = kmeans.predict(X_filtered)

print(np.unique(predictions, return_counts=True))


clustered = kmeans.fit_transform(X_filtered)