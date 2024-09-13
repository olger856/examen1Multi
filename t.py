# Import required libraries
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generar datos aleatorios con make_blobs
from sklearn.datasets import make_blobs

# Generamos 5000 puntos distribuidos en torno a cuatro centros específicos
X, y = make_blobs(n_samples=5000, centers=[[4, 4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)

# Visualizamos los datos generados
plt.scatter(X[:, 0], X[:, 1], marker='.')
plt.title('Datos generados aleatoriamente')
plt.show()

# Aplicamos K-means con 4 clusters
k_means = KMeans(init="k-means++", n_clusters=4, n_init=12)
k_means.fit(X)

# Obtenemos las etiquetas y los centros de los clusters
k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_

# Visualización del resultado del clustering
fig = plt.figure(figsize=(6, 4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(len(k_means_cluster_centers)), colors):
    my_members = (k_means_labels == k)
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

ax.set_title('KMeans Clustering con 4 clusters')
plt.show()

# Ahora aplicamos K-means con 3 clusters para comparar
k_means3 = KMeans(init="k-means++", n_clusters=3, n_init=12)
k_means3.fit(X)

fig = plt.figure(figsize=(6, 4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means3.labels_))))

ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(len(k_means3.cluster_centers_)), colors):
    my_members = (k_means3.labels_ == k)
    cluster_center = k_means3.cluster_centers_[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

ax.set_title('KMeans Clustering con 3 clusters')
plt.show()
