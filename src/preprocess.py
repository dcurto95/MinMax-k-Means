from sklearn.preprocessing import StandardScaler

def preprocess_dataset(X):
    return StandardScaler(with_mean=True, with_std=True).fit_transform(X)
