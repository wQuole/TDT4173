import numpy as np
from os import path
from math import sqrt, pow
from sklearn.model_selection import train_test_split

def euclidean_distance(v, u, length):
    distance = 0
    for x in range(length):
        distance += sqrt(pow((v[x] - u[x]), 2))
    return distance

def dataloder(filePath):
    data = np.genfromtxt(filePath, skip_header=True, delimiter=',')
    X = data[:, :-1]
    y = data[:, 3]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    return X_train, X_test, y_train, y_test

def get_neighbor(X_train, X_instance, k):
    return None

def predict(X_test):
    return None

def evaulate(y_test, y_pred):
    correct = 0
    for x in range(len(y_test)):
        if y_test[x] == y_pred[x]:
            correct += 1
    return (correct/len(y_test))*100

def main():
    knn_regression_data = path.abspath("dataset/knn_regression.csv")
    X_train, X_test, y_train, y_test = dataloder(knn_regression_data)
    print(f"Number of elements:\t{X_train.size}")
    print(f"Shape of the data:\t{np.shape(X_train)}")


if __name__ == '__main__':
    main()