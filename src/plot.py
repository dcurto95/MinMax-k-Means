import matplotlib.pyplot as plt

def plot_predictions(X,labels):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.show()