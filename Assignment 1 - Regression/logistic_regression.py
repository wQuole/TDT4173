from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def preprocess_data(filepath, transform=False):
    df = pd.read_csv(filepath, names=['x1', 'x2', 'y'], header=None)
    df.insert(0, 'x0', 1, True)  # Add new dimension because of merging bias into weight vector --> w0 = bias
    X, y = df.values[:, [0, 1, 2]], df.values[:, -1]
    if transform:
        X_t = []
        for row in X:
            #row = np.insert(row, 1, np.sqrt(2)*row[0]*row[1])
            X_t.append(np.power(row, 3))
        X = np.asarray(X_t)
    return X, y, df


class LogisticRegression:
    def __init__(self, eta, n, b=1):
        self.learning_rate = eta
        self.n_iterations = n
        self.bias = b

    def __str__(self):
        return (f"*** Logistc Regression ***\n"
                f"Learning rate:\t{self.learning_rate}\n"
                f"Iterations:\t{self.n_iterations}\n"
                f"Bias:\t{self.bias}")

    @staticmethod
    def sigmoid(z):
        return 1/(1 + np.exp(-z))

    @staticmethod
    def h(features, weights):
        return np.dot(features, weights)

    def predict(self, features, weights):
        return self.sigmoid(self.h(features, weights))

    @staticmethod
    def classify(prob):
        return 1 if prob >= 0.5 else 0

    def cross_entropy(self, features, weights, labels):
        y_pred = self.predict(features, weights)
        y = labels

        return np.mean(-y*np.log(y_pred) - (1-y)*np.log(1-y_pred))

    def gradient_descent(self, features, weights, labels):
        N = len(features)
        predictions = self.predict(features, weights)

        gradient = np.dot(features.T, predictions - labels)
        gradient /= N
        gradient *= self.learning_rate
        weights -= gradient

        return weights

    def fit(self, features, labels, weights):
        loss = []

        for i in range(self.n_iterations + 1):
            weights = self.gradient_descent(features, weights, labels)
            cost = self.cross_entropy(features, weights, labels)
            print(f"i\tCost: {cost}")
            loss.append(cost)

            if i % 500 == 0:
                print(f"Iteration: {i} weights:\n{weights}")

        return weights, loss

    def accuracy(self, predictions, actual):
        hits, misses = 0.0, 0.0
        for i in range(len(actual)):
            print(f"P(x) = {round(predictions[i], 5)}\t--> {self.classify(predictions[i])}\t--> Actually: {actual[i]}")
            if self.classify(predictions[i]) == actual[i]:
                hits += 1.0
            else:
                misses += 1.0
        print(f"Hits: {hits} | Misses: {misses}")
        return hits/len(actual)

    @staticmethod
    def divide_data(features, labels):
        positive, negative = [], []

        for i in range(len(labels)):
            if labels[i] == 1:
                positive.append(features[i])
            else:
                negative.append(features[i])

        return positive, negative

    def visualize_training_data(self, x, y):
        p, n = self.divide_data(x, y)

        plt.scatter([x[1] for x in p], [x[2] for x in p], label='positive', c='#1f77b4')
        plt.scatter([j[1] for j in n], [j[2] for j in n], label='negative', c='#ff7f0e')
        plt.show()


if __name__ == '__main__':
    # Linearly separable data
    train_data_1 = path.abspath('dataset/classification/cl_train_1.csv')
    test_data_1 = path.abspath('dataset/classification/cl_test_1.csv')
    # NOT Linearly separable data
    train_data_2 = path.abspath('dataset/classification/cl_train_2.csv')
    test_data_2 = path.abspath('dataset/classification/cl_test_2.csv')

    X_train, y_train, _ = preprocess_data(train_data_2, transform=True)
    X_test, y_test, _ = preprocess_data(test_data_2, transform=True)

    logit = LogisticRegression(0.03, 2000)
    pos, neg = logit.divide_data(X_train, y_train)
    logit.visualize_training_data(X_train, y_train)

    initial_weights = np.zeros(X_train.shape[1])
    weights, train_loss = logit.fit(X_train, y_train, initial_weights)
    _, test_loss = logit.fit(X_test, y_test, initial_weights)

    predictions = logit.predict(X_test, weights)
    print(logit)
    print("Accuracy:\n", logit.accuracy(predictions, y_test))

    plt.plot(train_loss, c='#1f77b4')
    plt.plot(test_loss, c='#ff7f0e')
    plt.legend(("Training loss", "Test loss"))
    plt.title("Training vs Test Loss")
    plt.show()

