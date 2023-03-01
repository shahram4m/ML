import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


np.random.seed(0)

# get 5000 xy around  4 center and define y for each point
x, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)

print(x)
print(y)

plt.scatter(x[:, 0], x[:, 1], marker='.')
plt.show()


# n_init = 12 : means 12 you can calc
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)

k_means.fit(x)

k_means_labels = k_means.labels_
k_means_labels


k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers