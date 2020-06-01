from sklearn.metrics import silhouette_score, calinski_harabasz_score,davies_bouldin_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
import numpy as np


def compute_metrics(labels_true, labels_pred, X):

    #Internal indices
    silhouette = silhouette_score(X, labels_pred)
    calinski = calinski_harabasz_score(X, labels_pred)
    dbi = davies_bouldin_score(X, labels_pred)

    #External indices
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    rand_score = adjusted_rand_score(labels_true, labels_pred)
    # ami = adjusted_mutual_info_score(labels_true, labels_pred)

    metrics = [silhouette, calinski, dbi, nmi, rand_score]
    return metrics



def compute_sum_intracluster_variance(X, cluster_centers, labels):
    sum_variance = 0
    for index, c in enumerate(np.unique(labels)):
        mask = np.where(labels == c)
        d = np.power(np.linalg.norm(X[mask, :] - cluster_centers[index]), 2)
        sum_variance = sum_variance + d
    return sum_variance

def compute_maximum_variance(X,cluster_centers,labels):
    maximum_variance = 0
    for index, c in enumerate(np.unique(labels)):
        mask = np.where(labels == c)
        d = np.power(np.linalg.norm(X[mask, :] - cluster_centers[index]), 2)
        if d > maximum_variance:
            maximum_variance = d
    return maximum_variance
