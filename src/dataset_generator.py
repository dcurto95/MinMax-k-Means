from sklearn.datasets import make_blobs


def generate_dataset(number_dataset=0):
    n_samples = 1500
    random_state = 170
    n_features = 2
    centers = 3

    X = None
    y = None

    if number_dataset == 0:
        # Different variance
        X, y = make_blobs(n_samples=n_samples,
                          cluster_std=[1.0, 2.5, 0.5],
                          random_state=random_state,
                          n_features=n_features,
                          centers=centers)
    else:
        X, y = make_blobs(n_samples=n_samples,
                          cluster_std=[1.8, 2.3, 1.9, 2, 2.6],
                          random_state=random_state,
                          n_features=n_features,
                          centers=5)
    return X, y
