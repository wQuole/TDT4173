import numpy as np
import pandas as pd
from os import path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def np_load_data(filePath):
    data = np.genfromtxt(filePath, skip_header=True, delimiter=',')
    X = data[:, :-1]
    y = data[:, 3]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test


def load_data(filePath):
    dataset = pd.read_csv(filePath, header=None)
    X_train = dataset[dataset.columns[:-1]]
    y = dataset[dataset.columns[-1]]
    return X_train, y, dataset


def euclidean_distance(candidate, data):
    distance = 0
    for i in range(len(candidate)):
        print(f"{candidate} --> {i}")
        distance += np.square(candidate[i] - data[i])
    return np.sqrt(distance)


def evaluate(y_test, y_pred):
    correct = 0
    for x in range(len(y_test)):
        if y_test[x] == y_pred[x]:
            correct += 1
    return (correct/len(y_test))*100


def pretty_print(data, head=False):
    if head:
        data = data[:10]
    for i in range(len(data)):
        print(f"{i}:\t {data[i]}")


def plurality_vote(all_neighbors):
    # returns the neighbors with most occurrences
    return np.argmax(np.bincount(all_neighbors))


def plotting(X, y, title, acc=1, threeDim=False):
    if not threeDim:
        X = X.drop(['x0'], axis=1)
        plt.plot(X, y, 'o', label='Original data', markersize=10)
        plt.title(title)
        plt.suptitle(f"Loss: {acc*100}%", va='top')
        plt.legend()
        plt.show()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        datapoints = ax.scatter(X[:,0], X[:,1], y, marker='x')
        plt.title(title)
        plt.suptitle(f"Loss: {acc * 100}%", va='top')
        ax.legend([datapoints], ['datapoints', 'predictions'], numpoints=1)

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')

        plt.show()

def kNN(X_train, test_Instance, k=10):
    distances = []
    voting_dict = {}
    for x in X_train:
        distances.append(euclidean_distance(test_Instance, X_train.iloc[x]))
    sorted = np.argsort(distances)[:10]


if __name__ == '__main__':
    knn_regression_data = path.abspath("dataset/knn_regression.csv")
    # X_train, X_test, y_train, y_test = np_load_data(knn_regression_data)
    X_train, y_train, dataset = load_data(knn_regression_data)
    print(type(X_train))
    # plotting(X_train, y_train, "Original data", threeDim=True)
    # print(f"Number of elements:\t{X_train.size}")
    # print(f"Shape of the data:\t{np.shape(X_train)}")
    # print(dataset.head())
    kNN(X_train, pd.DataFrame(X_train.iloc[124]), k=10)



