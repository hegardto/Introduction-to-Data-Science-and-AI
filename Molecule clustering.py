# -*- coding: utf-8 -*-

# -- 1 --

#Imports and df conversion
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('data_all.csv')
df = df.sort_values('phi', ascending=True)

#Scatter plot of distribution of phi and psi
xValues = df['phi']
yValues = df['psi']

plt.figure(figsize=(20,10))
plt.scatter(xValues, yValues)
plt.title('Distribution of phi and psi combinations')
plt.xlabel('Phi')
plt.ylabel('Psi')
plt.grid(color='black', linestyle='-', linewidth=0.1)
plt.savefig('Scatter')
plt.show()

#Heatmap for 12x12 different "buckets".
pd.set_option('precision', 0)
df1 = df.round(0)
df1['phi'] = pd.cut(df1['phi'], 12)
df1['psi'] = pd.cut(df1['psi'], 12)
count = df1.groupby(['phi','psi']).size().reset_index().rename(columns={0:'count'})
heatmap_data = pd.pivot_table(count, values='count', index='psi', columns='phi')
sns.heatmap(heatmap_data, cmap="rocket_r", linewidths=0.05, linecolor='black')
plt.savefig('Heatmap',dpi=100)

# -- 2 --

from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

#Create new df with phi and psi values
df2 = df[['phi','psi']]

#Create kMeans with 5 clusters for df2
kmeans = KMeans(n_clusters=5, random_state=0).fit(df2)
plt.figure(figsize=(20,10))

#Plot kmeans
plt.scatter(xValues,yValues)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x')
plt.title('Data points and cluster centroids')
plt.savefig('KMeans.png',dpi=400)
plt.show()

#For loop that calculates the square error for K values 1 to 10
cost =[] 
for i in range(1, 10): 
    KM = KMeans(n_clusters = i, max_iter = 500) 
    KM.fit(df2) 
    cost.append(KM.inertia_)      
  
# plot the square error against K values 
plt.plot(range(1, 10), cost, color ='black', linewidth ='3') 
plt.xlabel("Value of K") 
plt.ylabel("Squared Error (Sum)")
plt.grid(True)
plt.savefig('Elbow.png',dpi=400)
plt.show()

from sklearn.metrics import silhouette_score

#Calculate silhouette score for k between 2 and 5
km_scores= []
km_silhouette = []
for i in range(2,6):
    km = KMeans(n_clusters=i, random_state=0).fit(df2)
    preds = km.predict(df2)
    km_scores.append(-km.score(df2))
    
    silhouette = silhouette_score(df2,preds)
    km_silhouette.append(silhouette)
    print("Silhouette score for number of cluster(s) {}: {}".format(i,silhouette))

#Plot silhouette score for varying values of k
plt.figure(figsize=(7,4))
plt.title("Silhouette score for varying number of K",fontsize=16)
plt.scatter(x=[i for i in range(2,6)],y=km_silhouette,s=150,edgecolor='k')
plt.grid(True)
plt.xlabel("Number of clusters (K)",fontsize=14)
plt.ylabel("Silhouette score",fontsize=15)
plt.xticks([i for i in range(2,6)],fontsize=14)
plt.yticks(fontsize=15)
plt.savefig('Silhouette.png',dpi=400)
plt.show()

kmeans = KMeans(n_clusters=3, random_state=0).fit(df2)
plt.figure(figsize=(20,10))

plt.scatter(xValues,yValues)

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x')
plt.title('Data points and cluster centroids')
plt.savefig('KMeans2.png',dpi=400)
plt.xlabel('Phi')
plt.ylabel('Psi')
plt.show()

from sklearn.cluster import KMeans 

#Create a new Kmeans with optimal clusters = 3
ktest = KMeans(n_clusters=3) 
#fitting the model to X 
ktest.fit(df2) 
plt.figure(figsize=(20,10))
plt.title('Data points and cluster by color')
plt.xlabel('Phi')
plt.ylabel('Psi')

#predicting labels (y_pred)
y_pred = ktest.predict(df2)
plt.scatter(df2['phi'], df2['psi'], c=y_pred, cmap=plt.cm.Paired) 
plt.savefig('KMeansC.png',dpi=400)

plt.show()

#Create dataframe df3 and add cos and sin values for all phi and psi angles
df3 = df
df3['phi_cos'] = np.cos(df3['phi'] * np.pi / 180)
df3['phi_sin'] = np.sin(df3['phi'] * np.pi / 180)
df3['psi_cos'] = np.cos(df3['psi'] * np.pi / 180)
df3['psi_sin'] = np.sin(df3['psi'] * np.pi / 180)

#Drop all columns except cos and sin values, and create a new KMeans, with optimal clusters 3
df3 = df3[['phi_cos','phi_sin','psi_cos','psi_sin']]
ktest = KMeans(n_clusters=3)

#Fit the model to df3
ktest.fit(df3)

#Plot the datapoints by prediciting the label for each point based on the cos and sin values respecitvely
plt.figure(figsize=(20,10))
plt.title('Data points and cluster by color')
plt.xlabel('Phi')
plt.ylabel('Psi')
y_pred = ktest.predict(df3)
plt.scatter(df2['phi'], df2['psi'], c=y_pred, cmap=plt.cm.Paired) 
plt.savefig('KMeansC2.png',dpi=400)

plt.show()

# -- 3 --

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Two differents standardized sets, Y for plotting and X for computing
Y = StandardScaler().fit_transform(df[['phi','psi']])
X = StandardScaler().fit_transform(df3[['phi_cos','phi_sin','psi_cos','psi_sin']])

# Compute DBSCAN
db = DBSCAN(eps=0.35, min_samples=80, metric = 'euclidean').fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Calculate number of labels and removing noise
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

# Print results
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

# Create a nearest neighbor classifier with k neighbors
k = 80
neigh = NearestNeighbors(n_neighbors=k)

# Fit to X and save distances and their indices
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)

# Create Elbow graph by computing average distance to every datapoint's k nearest neighbors
meanDist = []
count = 0
for o in distances:
    meanDis = 0
    for i in range (1,k):
        meanDis = meanDis + distances[count,i]
    meanDist.append(meanDis/k)
    count = count + 1
meanDist  = np.sort(meanDist, axis=0)
plt.figure(figsize=(20,10))
plt.grid(True)
plt.plot(meanDist)
plt.savefig('Elbow_DBSCAN.png',dpi=400)
plt.show()

# Plot the clusters calculated by the DBSCAN model in different colors in a standardized scale
unique_labels = set(labels)
plt.figure(figsize=(20,10))
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = Y[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = Y[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.savefig('Clusters_DBSCAN.png',dpi=400)
plt.show()

# Calculate number of every residue name belonging to the noise and plot in bar chart
df4 = df
df4['label'] = db.labels_
df4 = df4.loc[df4['label'] == -1]
df5 = df4.groupby('residue name').size().sort_values(ascending=False).reset_index(name='label')
df5.plot.bar(x="residue name", y="label", rot=70, title="Number of outliers per amino acid residue type")
plt.savefig('Bar.png',dpi=400)
plt.show(block=True)

# Calculate number of clusters for different min_samples
clusters = []
count = 1
for i in range (1,60):
    db = DBSCAN(eps=0.35, min_samples=i, metric = 'euclidean').fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    clusters.append(n_clusters_)
    print (count)
    count = count + 1

# Plot the number of clusters together with the different min_samples
plt.figure(figsize=(20,10))
plt.title("Number of clusters when varying the minimum amount of neighbors")
plt.grid(True)
plt.xlabel('Value of minimum amount of neighbors')
plt.ylabel('Number of clusters')
plt.plot(clusters)
plt.savefig('Different#ofNeighs.png',dpi=400)
plt.show()

# Calculate number of clusters for different eps
clusters = []
count = 1
eps = np.arange(0.01, 0.7, 0.01)
for i in eps:
    db = DBSCAN(eps=i, min_samples=80, metric = 'euclidean').fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    clusters.append(n_clusters_)
    print (count)
    count = count + 1

# Plot the number of clusters together with the different eps
plt.figure(figsize=(20,10))
plt.title("Number of clusters when varying the minimum amount of neighbors")
plt.grid(True)
plt.xlabel('Value of eps')
plt.ylabel('Number of clusters')
plt.plot(eps,clusters)
plt.savefig('DifferentEps.png',dpi=400)
plt.show()

# -- 4 --

#Create dataframe df6 and add cos and sin values for all phi and psi angles
df6 = df 
df6['phi_cos'] = np.cos(df6['phi'] * np.pi / 180)
df6['phi_sin'] = np.sin(df6['phi'] * np.pi / 180)
df6['psi_cos'] = np.cos(df6['psi'] * np.pi / 180)
df6['psi_sin'] = np.sin(df6['psi'] * np.pi / 180)

#Keep rows with residue name = PRO and drop all columns except sin and cos values
df6 = df6.loc[df6['residue name'] == 'PRO']

df6 = df6[['phi_cos','phi_sin','psi_cos','psi_sin']]

df7 = df
df7 = df7.loc[df7['residue name'] == 'PRO']

#Standardize axis 
Y = StandardScaler().fit_transform(df7[['phi','psi']])
X = StandardScaler().fit_transform(df6[['phi_cos','phi_sin','psi_cos','psi_sin']])

# Compute DBSCAN
db = DBSCAN(eps=0.4, min_samples=8, metric = 'euclidean').fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

#Calculate the 8 nearest neighbors
k = 8
neigh = NearestNeighbors(n_neighbors=k)

#Create list with distnaces for neighbors
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)

#Calculate the mean distance to k nearest neighbors
meanDist = []
count = 0
for o in distances:
    meanDis = 0
    for i in range (1,k):
        meanDis = meanDis + distances[count,i]
    meanDist.append(meanDis/k)
    count = count + 1
meanDist  = np.sort(meanDist, axis=0)

#Plot elbow dbscan for selected K value, in order to select optimal epsilon
plt.figure(figsize=(20,10))
plt.grid(True)
plt.plot(meanDist)
plt.savefig('Elbow_DBSCAN.png',dpi=400)
plt.show()

# Plot the clusters calculated by the DBSCAN model in different colors in a standardized scale

unique_labels = set(labels)
plt.figure(figsize=(20,10))
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = Y[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = Y[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.savefig('Clusters_DBSCAN2.png',dpi=400)
plt.show()

#Create dataframe df6 and add cos and sin values for all phi and psi angles
df8 = df
df8['phi_cos'] = np.cos(df8['phi'] * np.pi / 180)
df8['phi_sin'] = np.sin(df8['phi'] * np.pi / 180)
df8['psi_cos'] = np.cos(df8['psi'] * np.pi / 180)
df8['psi_sin'] = np.sin(df8['psi'] * np.pi / 180)
df8 = df8.loc[df8['residue name'] == 'GLY']

#Drop all columns except cos and sin values, and create a new KMeans, with optimal clusters 3
df8 = df8[['phi_cos','phi_sin','psi_cos','psi_sin']]

df9 = df
df9 = df9.loc[df9['residue name'] == 'GLY']

Y = StandardScaler().fit_transform(df9[['phi','psi']])
X = StandardScaler().fit_transform(df8[['phi_cos','phi_sin','psi_cos','psi_sin']])

# Compute DBSCAN
db = DBSCAN(eps=0.38, min_samples=8, metric = 'euclidean').fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

#Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

#Calculate the 8 nearest neighbors
k = 8
neigh = NearestNeighbors(n_neighbors=k)

#Create list with distnaces for neighbors
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)

#Calculate the mean distance to k nearest neighbors
meanDist = []
count = 0
for o in distances:
    meanDis = 0
    for i in range (1,k):
        meanDis = meanDis + distances[count,i]
    meanDist.append(meanDis/k)
    count = count + 1
meanDist  = np.sort(meanDist, axis=0)

#Plot elbow dbscan for selected K value, in order to select optimal epsilon
plt.figure(figsize=(20,10))
plt.grid(True)
plt.plot(meanDist)
plt.savefig('Elbow_DBSCAN.png',dpi=400)
plt.show()

# Plot the clusters calculated by the DBSCAN model in different colors in a standardized scale
unique_labels = set(labels)
plt.figure(figsize=(20,10))
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = Y[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = Y[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.savefig('Clusters_DBSCAN3.png',dpi=400)
plt.show()

