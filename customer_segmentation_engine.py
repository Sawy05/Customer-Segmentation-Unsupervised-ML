import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import skfuzzy as fuzz

# K-Medoids from pyclustering
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils.metric import distance_metric, type_metric

# Load dataset
df = pd.read_csv("marketing_campaign.csv", sep="\t")

# Basic cleaning
df = df.dropna()
df['Age'] = 2025 - df['Year_Birth']
df['Total_Spending'] = df[['MntWines','MntFruits','MntMeatProducts',
                           'MntFishProducts','MntSweetProducts','MntGoldProds']].sum(axis=1)
df['Children'] = df['Kidhome'] + df['Teenhome']

# Select numerical features
num_cols = ['Income','Age','Recency','Total_Spending','Children']
X = df[num_cols]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------
# 1. Hierarchical Clustering
# -----------------------------------
hier = AgglomerativeClustering(n_clusters=4)
df['HierCluster'] = hier.fit_predict(X_scaled)

# -----------------------------------
# 2. K-Medoids (Using pyclustering)
# -----------------------------------
# Random initial medoids
import random
initial_medoids = random.sample(range(len(X_scaled)), 4)

# Define metric
metric = distance_metric(type_metric.EUCLIDEAN)

# Run K-Medoids
kmed = kmedoids(X_scaled, initial_medoids, metric=metric)
kmed.process()

clusters = kmed.get_clusters()
medoids = kmed.get_medoids()

# Convert cluster lists → label column
labels_kmed = [None] * len(X_scaled)
for cluster_id, cluster_points in enumerate(clusters):
    for point in cluster_points:
        labels_kmed[point] = cluster_id

df["KMedoidsCluster"] = labels_kmed

# -----------------------------------
# 3. Fuzzy C-Means (skfuzzy)
# -----------------------------------
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    X_scaled.T, 4, 2, error=0.005, maxiter=1000
)

df[[f"Fuzzy_{i}" for i in range(4)]] = u.T

# -----------------------------------
# Save Results
# -----------------------------------
df.to_csv("clustered_output.csv", index=False)
print("Processing complete. Outputs saved.")
