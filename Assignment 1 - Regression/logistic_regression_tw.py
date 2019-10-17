import math
import numpy as np
import csv_reader
import matplotlib.pyplot as plt
from operator import add
import matplotlib.colors as c


def predict(weights, x, b = False):
    if b:
        x = np.append(x,[x[0]**2, x[1]**2])
        sigma = lambda z:1/(1+math.e**(-z))
        z = np.dot(weights, np.matrix.transpose(np.array(x)))
        if sigma(z) >= 0.5:
            return 1
        else: return 0

    sigma = lambda z:1/(1+math.e**(-z))
    z = np.dot(weights, np.matrix.transpose(np.array(x)))
    return sigma(z)


def update_weights(weights, features, y, learning_rate):
    predictions = [predict(weights, features[i]) for i in range(len(features))]
    cumulative = [(predictions[0] - y[0])*x for x in features[0]]

    for i in range(1, len(y)):
        error = predictions[i] - y[i]
        a = [error* x for x in features[i]]
        cumulative = list(map(add, cumulative, a))
    return np.subtract(weights, [learning_rate*c for c in cumulative]), predictions


def crossentropy_error(predictions, y):
    ce = 0
    for i in range(len(predictions)):
        ce += -(y[i] * math.log(predictions[i]) + (1-y[i]) * math.log(1-predictions[i]))
    return ce/len(y)


def fit(x, y, iterations, x_test, y_test):
    weights = [0.5]*len(x[0])
    history = []
    history_test = []

    for n in range(iterations):
        weights, predictions = update_weights(weights, x, y, 0.05)
        p = [predict(weights, x) for x in x_test]
        history_test.append(crossentropy_error(p, y_test))
        history.append(crossentropy_error(predictions, y))
    return weights, history, history_test


def main():
    from os import path
    train_2 = path.abspath("dataset/classification/cl_train_2.csv")
    test_2 = path.abspath("/dataset/classification/cl_test_2.csv")
    data = csv_reader.load_data(train_2, None)
    test = csv_reader.load_data(test_2, None)
    #extract features and targets from datasets

    x = [x[0:len(x)-1] for x in data]
    for i in range(len(x)):
        x[i].append(x[i][0]**2)
        x[i].append(x[i][1]**2)

    y = [y[-1] for y in data]
    x_test = [x[0:len(x)-1] for x in test]
    for i in range(len(x_test)):
        x_test[i].append(x_test[i][0]**2)
        x_test[i].append(x_test[i][1]**2)
    y_test = [y[-1] for y in test]


    #fit and update_weights together make up the gradient descent optimization
    weights, history, history_test = fit(x, y, 1500, x_test, y_test)

    #plot the cross entropy error from each itaration
    plt.plot(history)
    plt.plot(history_test)
    plt.legend(("training error", "test error"))
    plt.show()

    #make predictions and separate into positive and negative examples for plots
    #predictions = [predict(weights, f) for f in x]
    xp = []
    xn = []
    for p in range(len(y)):
        if y[p] == 1:
            xp.append(x[p])
        else:
            xn.append(x[p])

    plt.plot([x[0] for x in xp], [x[1] for x in xp], '+',label = "Target = 1")
    plt.plot([x[0] for x in xn], [x[1] for x in xn], 'o',label = "Target = 0")

    #generate decision Boundary points
    #x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    #y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    X = np.array(x)
    x_min, x_max = X[:, 2].min() - .5, X[:, 2].max() + .5
    y_min, y_max = X[:, 3].min() - .5, X[:, 3].max() + .5
    h = .008  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z = []
    for n in np.c_[xx.ravel(), yy.ravel()]:
        z.append(predict(weights, n, b=True))

    Z = np.array(z)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=c.ListedColormap(['#FF0000', '#32CD32']))

    # Plot also the training points
    plt.plot([x[0] for x in xp], [x[1] for x in xp], '+',c = "blue",label = "Target = 1")
    plt.plot([x[0] for x in xn], [x[1] for x in xn], 'o',c = 'blue',label = "Target = 0")


    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


    #plx =[min(x[:]), max(x[:])]
    #ply = [min(x[:]), max(x[:])]
    #plt.plot(x1, x2)
    #plt.show()

main()
