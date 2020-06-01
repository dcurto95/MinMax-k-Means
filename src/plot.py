import matplotlib.pyplot as plt
from dataset_generator import  retrieve_dataset

def plot_predictions(X,labels):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.show()

def plot_artificial_datasets():
    X,y = retrieve_dataset('artificial_0')
    X_1, y_1 = retrieve_dataset('artificial_1')
    X_2, y_2 = retrieve_dataset('artificial_2')

    plt.figure()
    plt.subplot(131)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title("Artificial_0 - Low variance")
    plt.subplot(132)
    plt.scatter(X_1[:, 0], X_1[:, 1], c=y_1)
    plt.title("Artificial_1 - High variance")
    plt.subplot(133)
    plt.scatter(X_2[:, 0], X_2[:, 1], c=y_2)
    plt.title("Artificial_2 - Medium variance")
    plt.show()