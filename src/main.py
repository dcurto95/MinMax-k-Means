import pandas as pd
import numpy as np
from _min_max_kmeans import MinMaxKMeans
from improved_min_max_kmeans import ImprovedMinMaxKMeans
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import time
n_samples = 1500
random_state = 170


# Different variance
X_varied, y_varied = make_blobs(n_samples=n_samples,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=random_state)

start = time.time()
classifier = ImprovedMinMaxKMeans(n_clusters=3)
classifier.fit(X_varied)
end = time.time()
elapsed = end - start
print("Improved", elapsed)

plt.figure()
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=classifier.labels_)
plt.title("Incorrect Number of Blobs")
plt.show()
import sys
sys.exit()


plt.figure(figsize=(12, 12))

n_samples = 1500
random_state = 170
n_features = 4
X, y = make_blobs(n_samples=n_samples, random_state=random_state, n_features=n_features)

start = time.time()
classifier = ImprovedMinMaxKMeans(n_clusters=3)
classifier.fit(X)
end = time.time()
elapsed = end -start
print("Improved", elapsed)


import sys
sys.exit()

start = time.time()
min_max = MinMaxKMeans(n_clusters=3)
min_max.fit(X)
end = time.time()

elapsed = end -start
print("Original", elapsed)
plt.subplot(211)
plt.scatter(X[:, 0], X[:, 1], c=min_max.labels_)
plt.title("Incorrect Number of Blobs")


# start = time.time()
# # Incorrect number of clusters
# y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X)
# end = time.time()
#
# elapsed = end -start
# print(elapsed)

plt.subplot(212)
plt.scatter(X[:, 0], X[:, 1], c=classifier.labels_)
plt.title("Incorrect Number of Blobs")
plt.show()

import sys
sys.exit()

# Anisotropicly distributed data
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation)
min_max = MinMaxKMeans(n_clusters=3)
min_max.fit(X_aniso)

#y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)

plt.subplot(222)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=min_max.labels_)
plt.title("Anisotropicly Distributed Blobs")


# Different variance
X_varied, y_varied = make_blobs(n_samples=n_samples,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=random_state)
#y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)

min_max = MinMaxKMeans(n_clusters=3)
min_max.fit(X_varied)

plt.subplot(223)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=min_max.labels_)
plt.title("Unequal Variance-MinMax")

# plt.subplot(224)
# plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
# plt.title("Unequal Variance-Kmeans")


# Unevenly sized blobs
X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))

min_max = MinMaxKMeans(n_clusters=3)
min_max.fit(X_filtered)

# y_pred = KMeans(n_clusters=3,
#                 random_state=random_state).fit_predict(X_filtered)

plt.subplot(224)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=min_max.labels_)
plt.title("Unevenly Sized Blobs")


plt.show()
