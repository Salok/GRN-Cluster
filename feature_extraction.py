import pandas as pd
import pyarrow.parquet as pq
import numpy as np

"""Scratch file to take the important features from the full brain model and take the data from the full brain dataset
 for clustering afterwards. Take only the connected genes from the model."""

graph = pq.read_table("/home/cig/Desktop/global.gzip")
graph = graph.to_pandas()
graph.drop('Unnamed: 0', axis=1, inplace=True)

values = graph.values
mask = values != 0
rows = set([i for i in range(values.shape[0]) if np.any(mask[i])])
columns = set([i for i in range(values.shape[0]) if np.any(mask[:][i])])

keep = list(rows.union(columns))

connected_genes = graph.iloc[:, keep].columns
hubs_score = pd.DataFrame(data = [np.sum(graph.iloc[:, keep].values != 0, axis=0).tolist()], columns=connected_genes)

full_brain_data = pd.read_csv("/home/cig/Desktop/Allen Brain/data/clean_full_brain.csv")

clustering_dataset = full_brain_data[connected_genes]

clustering_dataset.to_csv("/home/cig/Desktop/PHC/full_feature_extraction_dataset.csv", index=False)
hubs_score.to_csv("/home/cig/Desktop/PHC/hubs_scores.csv", index=False)