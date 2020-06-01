import numpy as np
from sklearn.datasets import make_blobs
from improved_min_max_kmeans import ImprovedMinMaxKMeans
import matplotlib.pyplot as plt

#Toy dataset
X,y = make_blobs(n_samples=300,
                random_state=1,
                n_features=2,
                centers=3)
n_clusters = len(np.unique(y))

improved_minmax = ImprovedMinMaxKMeans(n_clusters=n_clusters, beta=0.3, verbose=0)
labels_minmax = improved_minmax.fit_predict(X)


plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels_minmax)
plt.title("Cluster assignments")
plt.show()



