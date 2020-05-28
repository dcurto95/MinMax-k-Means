import numpy as np
from copy import deepcopy
class ImprovedMinMaxKMeans:
    def __init__(self, n_clusters=8, p_max=0.5, p_step=0.01, beta=0.1, variance_threshold=10**-6, max_iter=500, verbose=0, n_init=10):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.p_max = p_max
        self.p_step = p_step
        self.beta = beta
        self.variance_threshold = variance_threshold
        self.max_iter = max_iter
        self.verbose = verbose
        self.labels_ = None
        self.cost_ = 0
        self.clusters_variance_ = None
        self.cluster_centers_ = None

    def fit(self, X):
        #TODO: Create the main code in a function
        #Validate the parameters (TODO: To Complete)
        self.validate_parameters()
        #Initialize cluster centroids
        self.cluster_centers_ = self.initialize_centroids(X)

        #Initialize cluster weights
        current_weights = [1/self.n_clusters] * self.n_clusters
        old_weights = [1 / self.n_clusters] * self.n_clusters
        #Initialize cluster assignments
        current_cluster_assignments = [[] for _ in range(self.n_clusters)]
        old_cluster_assignments = [[] for _ in range(self.n_clusters)]

        t = 0
        p_init = 0
        empty_cluster = False
        p = p_init
        converged = False
        while t < self.max_iter and not converged:
            t = t + 1
            current_cluster_assignments = self.update_cluster_assignments(X, current_weights, p)
            #Check for empty cluster and update its value
            if self.exists_singleton_cluster(current_cluster_assignments):
                empty_cluster = True
                p = p - self.p_step
                if p < p_init:
                    return None
                #Revert to the assignments and weights corresponding to the reduced p
                current_cluster_assignments = deepcopy(old_cluster_assignments)
                current_weights = old_weights.copy()

            #Update cluster centers
            self.update_cluster_centers(current_cluster_assignments, X)

            if p < self.p_max and not empty_cluster:
                #Store the current assignments in delta(p)
                old_cluster_assignments = deepcopy(current_cluster_assignments)
                #Store the previous weights in vector W(p)
                old_weights = np.copy(current_weights)
                p = p + self.p_step

            #Update the weights
            self.update_weights(current_weights, current_cluster_assignments, X, p)
            #Check for convergence
            cost = self.compute_cost(current_weights)
            converged = np.abs(cost - self.cost_) < self.variance_threshold
            # print("Iteration", t)
            # print("Diff variancce", np.abs(cost - self.cost_))
            self.cost_ = cost

        self.labels_ = self.get_instances_labels(current_cluster_assignments, X)


    def compute_cost(self, weights):
        cost = 0
        for k in range(self.n_clusters):
            cost = cost + weights[k]*self.clusters_variance_[k]
        return cost

    def get_instances_labels(self, cluster_assignments, X):
        N = X.shape[0]
        labels = np.zeros(N)
        for cluster, values in enumerate(cluster_assignments):
            labels[values] = cluster
        return labels

    def update_cluster_centers(self, current_cluster_assignments, X):
        #Update cluster centers
        for i in range(self.n_clusters):
            mask = (current_cluster_assignments[i])
            multiplication = np.sum(X[mask,:], axis=0)
            count = len(mask)
            self.cluster_centers_[i] = multiplication/count if count > 0 else 0

    def update_weights(self, weights, cluster_assignments, X, p):
        self.clusters_variance_ = self.compute_clusters_variance(cluster_assignments, X)
        total_variance = np.sum(np.power(self.clusters_variance_, 1/(1-p)))
        for k in range(self.n_clusters):
            variance = self.clusters_variance_[k]
            weights[k] = self.beta * weights[k] + (1 - self.beta)*np.power(variance, 1/(1-p))/total_variance


    def compute_clusters_variance(self, cluster_assignments,X):
        #Initialize variance
        variance = np.zeros(self.n_clusters)
        for k in range(self.n_clusters):
            mask = (cluster_assignments[k])
            variance[k] = np.power(np.linalg.norm(X[mask, :] - self.cluster_centers_[k]), 2)
        return variance


    def exists_singleton_cluster(self, cluster_assignments):
        #Return true if some cluster is empty or only one instance is found

        for cluster in cluster_assignments:
            if len(cluster) <= 1:
                return True
        return False

    def update_cluster_assignments(self, X, weights, p):
        # Update the cluster assignments
        new_clusters = [[] for _ in range(self.n_clusters)]
        N = X.shape[0]
        for i in range(N):
            cluster_index = self.compute_minimization_step(weights, X[i, :], p)
            new_clusters[cluster_index].append(i)
        return new_clusters


    def compute_minimization_step(self, weights, instance, p):
        distances = []
        for i in range(self.n_clusters):
            distance = np.power(weights[i], p) * np.power(np.linalg.norm(instance - self.cluster_centers_[i]), 2)
            distances.append(distance)
        return np.argmin(distances)


    def validate_parameters(self):
        if self.n_init <= 0:
            raise ValueError("Invalid number of initializations."
                             " n_init=%d must be bigger than zero." % self.n_init)
        if self.max_iter <= 0:
            raise ValueError(
                'Number of iterations should be a positive number,'
                ' got %d instead' % self.max_iter
            )

    def predict(self, X):
        return self.labels_

    def fit_predict(self, X):
        return self.fit(X).labels_


    def initialize_centroids(self, X):
        """
        Selects the initial centroids (randomly)
        :param X: The data where the points will be selected
        :param n_clusters: Number of initial points to select (cluster centers)
        """
        centroids_indexs = np.random.choice(range(len(X)), self.n_clusters, replace=False)
        return X[centroids_indexs]



