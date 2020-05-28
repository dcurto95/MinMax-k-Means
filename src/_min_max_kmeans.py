import numpy as np

class MinMaxKMeans:
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
        #Validate the parameters (TODO: To Complete)
        self.validate_parameters()
        #Initialize cluster centroids
        self.cluster_centers_ = self.initialize_centroids(X)

        #Initialize cluster weights
        current_weights = [1/self.n_clusters] * self.n_clusters
        old_weights = [1 / self.n_clusters] * self.n_clusters
        #Initialize cluster assignments
        current_cluster_assignments = np.zeros((X.shape[0], self.n_clusters))
        old_cluster_assignments = np.zeros((X.shape[0], self.n_clusters))

        t = 0
        p_init = 0
        empty_cluster = False
        p = p_init
        converged = False
        while t < self.max_iter and not converged:  #current_variance - previous_variance >= self.variance_stop
            t = t + 1
            self.update_cluster_assignments(X, current_cluster_assignments, current_weights, p)
            #Check for empty cluster and update its value
            if self.exists_singleton_cluster(current_cluster_assignments):
                empty_cluster = True
                p = p - self.p_step
                if p < p_init:
                    return None
                #Revert to the assignments and weights corresponding to the reduced p
                self.revert_assignments(current_cluster_assignments, current_weights, old_cluster_assignments, old_weights)

            #Update cluster centers
            self.update_cluster_centers(current_cluster_assignments, X)

            if p < self.p_max and not empty_cluster:
                #Store the current assignments in delta(p)
                old_cluster_assignments = np.copy(current_cluster_assignments)
                #Store the previous weights in vector W(p)
                old_weights = np.copy(current_weights)
                p = p + self.p_step

            #Update the weights
            self.update_weights(current_weights, current_cluster_assignments, X, p)
            #Check for convergence
            cost = self.compute_cost(current_weights)

            # print("Old cost", self.cost_)
            # print("New cost", cost)
            # print("Difference", np.abs(cost - self.cost_))
            # if np.abs(cost - self.cost_) < self.variance_threshold:
            #     converged = True
            converged = np.abs(cost - self.cost_) < self.variance_threshold
            self.cost_ = cost
        self.labels_ = self.get_instances_labels(current_cluster_assignments)


    def compute_cost(self, weights):
        cost = 0
        for k in range(self.n_clusters):
            cost = cost + weights[k]*self.clusters_variance_[k]
        return  cost

    def get_instances_labels(self, cluster_assignments):
        labels = np.zeros(cluster_assignments.shape[0])
        for k in range(self.n_clusters):
            indexes = np.argwhere(cluster_assignments[:, k] > 0)
            labels[indexes] = k
        return labels



    def update_cluster_centers(self, current_cluster_assignments, X):
        #Update cluster centers
        for i in range(self.n_clusters):
            mask = (current_cluster_assignments[:,i] > 0)
            multiplication = np.sum(X[mask,:], axis=0)
            count = np.sum(current_cluster_assignments[:,i], axis=0)
            self.cluster_centers_[i] = multiplication/count

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
            mask = (cluster_assignments[:, k] > 0)
            variance[k] = np.power(np.linalg.norm(X[mask, :] - self.cluster_centers_[k]), 2)
        return variance

    def revert_assignments(self, current_cluster_assignments, current_weights, old_cluster_assignments, old_weights):
        N, K = current_cluster_assignments.shape
        for i in range(N):
            for k in range(K):
                current_cluster_assignments[i, k] = old_cluster_assignments[i, k]
                current_weights[k] = old_weights[k]


    def exists_singleton_cluster(self, cluster_assignments):
        #Return true if some cluster is empty or only one instance is found
        count_sum = np.sum(cluster_assignments, axis=0)
        for count in count_sum:
            if count <= 1:
                return True
        return False

    def update_cluster_assignments(self, X, cluster_assignments, weights, p):
        # Update the cluster assignments
        N = X.shape[0]
        for i in range(N):
            for k in range(self.n_clusters):
                if k == self.compute_minimization_step(k, weights, X[i,:], p):
                    cluster_assignments[i, k] = 1
                else:
                    cluster_assignments[i, k] = 0

    def compute_minimization_step(self, k, weights, instance, p):
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



