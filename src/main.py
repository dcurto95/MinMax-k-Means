import numpy as np
from _min_max_kmeans import MinMaxKMeans
from improved_min_max_kmeans import ImprovedMinMaxKMeans
from sklearn.cluster import KMeans
from src import metrics, preprocess
from sklearn import datasets
import time
import dataset_generator


X, y = datasets.load_iris(return_X_y=True)
X,y = datasets.load_breast_cancer(return_X_y=True)
X,y = datasets.load_wine(return_X_y=True)
X = preprocess.preprocess_dataset(X)


X,y = dataset_generator.generate_dataset(0)
n_clusters = len(np.unique(y))
minmax = MinMaxKMeans(n_clusters=n_clusters, verbose=1, max_iter=100)
labels = minmax.fit_predict(X)
cost = minmax.cost_
inertia = metrics.compute_sum_intracluster_variance(X, minmax.cluster_centers_, labels)

kmeans = KMeans(n_clusters=n_clusters, init='random')
labels = kmeans.fit_predict(X)
cost_kmeans = metrics.compute_maximum_variance(X, kmeans.cluster_centers_, labels)


import plot
plot.plot_predictions(X, labels)




start = time.time()
improved_MinMax = ImprovedMinMaxKMeans(n_clusters=n_clusters, verbose=0)
labels = improved_MinMax.fit_predict(X)
end = time.time()
elapsed = end - start


start = time.time()
minmax = MinMaxKMeans(n_clusters=n_clusters, verbose=0)
labels = minmax.fit_predict(X)
end = time.time()
elapsed = end - start


start = time.time()
kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=1, max_iter=500)
labels = KMeans.fit_predict(X)
end = time.time()
elapsed = end - start

start = time.time()
kmeans_init = KMeans(n_clusters=n_clusters, init='k-means++')
labels = KMeans.fit_predict(X)
end = time.time()
elapsed = end - start




