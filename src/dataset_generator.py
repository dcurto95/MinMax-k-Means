from sklearn.datasets import make_blobs
from sklearn import datasets

def retrieve_dataset(dataset_name):
    n_samples = 500
    random_state = 170
    n_features = 2
    centers = 3

    if dataset_name == 'iris':
        return datasets.load_iris(return_X_y=True)

    elif dataset_name =='breast_cancer':
        return datasets.load_breast_cancer(return_X_y=True)

    elif dataset_name == 'wine':
        return datasets.load_wine(return_X_y=True)

    elif dataset_name == 'artificial_0':
        return make_blobs(n_samples=400,
                          random_state=random_state,
                          cluster_std=[1.1, 1.1, 1.1],
                          n_features=n_features,
                          centers=[[-5, 0], [7, -5], [7, 8]])

    elif dataset_name == 'artificial_1':
        # Different variance
        return make_blobs(n_samples=600,
                          cluster_std=[1.0, 1.8, 1.2],
                          random_state=random_state,
                          n_features=n_features,
                          centers=centers)

    elif dataset_name == 'artificial_2':
        return make_blobs(n_samples=700,
                          cluster_std=[1.5, 1.5, 1.5, 1.5, 1.5],
                          random_state=random_state,
                          n_features=n_features,
                          centers=5)


    else:
        return make_blobs(n_samples=n_samples,
                random_state=random_state,
                n_features=n_features,
                centers=centers)