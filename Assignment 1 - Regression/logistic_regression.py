from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import logistic


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

    def preprocess_data(self, filepath):
        df = pd.read_csv(filepath, names=['x1', 'x2', 'y'], header=None)
        df.insert(0, 'x0', self.bias, True)  # Add new dimension because of merging bias into weight vector --> w0 = bias
        X, y = df.values[:, [0, 1, 2]], df.values[:, -1]
        return X, y, df

    @staticmethod
    def sigmoid(z):
        return 1/(1 + np.exp(-z))

    @staticmethod
    def h(features, weights):
        return np.dot(features, weights)

    def predict(self, features, weights):
        return self.sigmoid(self.h(features, weights))

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

    def classify(self, predictions):
        decisionBoundary = lambda prob: 1 if prob >= 0.5 else 0
        return np.vectorize(decisionBoundary(predictions))

    def fit(self, features, labels, weights):
        training_loss = []

        for i in range(self.n_iterations+1):
            weights = self.gradient_descent(features, weights, labels)
            cost = self.cross_entropy(features, weights, labels)
            print(f"i\tCost: {cost}")
            training_loss.append(cost)

            if i % 1000 == 0:
                print(f"Iteration: {i} weights:\n{weights}")

        return weights, training_loss

    def accuracy(self, predictions, actual):
        diff = predictions - actual
        return 1.0 - diff/len(diff)

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
        plt.scatter([j[1] for j in neg], [j[2] for j in neg], label='negative', c='#ff7f0e')
        plt.show()

    def plot_decision_boundary(self, pos, neg):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter([x[1] for x in pos], [x[2] for x in pos], s=35, c='b', marker="o", label='Trues')
        ax.scatter([j[1] for j in neg], [j[2] for j in neg], s=25, c='r', marker="s", label='Falses')

        plt.legend(loc='upper right');
        ax.set_title("Decision Boundary")
        ax.set_xlabel('N/2')
        ax.set_ylabel('Predicted Probability')
        plt.axvline(.5, color='black')
        plt.show()


if __name__ == '__main__':
    train_data = path.abspath('dataset/classification/cl_train_1.csv')
    test_data = path.abspath('dataset/classification/cl_test_1.csv')

    logit = LogisticRegression(0.05, 1000)
    X_train, y_train, _ = logit.preprocess_data(train_data)
    X_test, y_test, _ = logit.preprocess_data(test_data)

    #pos, neg = logit.divide_data(X,y)
    #logit.visualize_training_data(X, y)

    initial_weights = np.zeros(X_train.shape[1])
    weights, train_loss = logit.fit(X_train, y_train, initial_weights)
    _, test_loss = logit.fit(X_test, y_test, weights)
    print(logit)
    plt.plot(train_loss, c='#1f77b4')
    plt.plot(test_loss, c='#ff7f0e')
    plt.legend(("Training loss", "Test loss"))
    plt.title("Training vs Test Loss")
    plt.show()
    # sklearnLogReg = logistic.LogisticRegression()
    # a = sklearnLogReg.fit(X, y)
    # print(f"Sklearn-LogReg()-weights:\n{a.coef_} after {sklearnLogReg.max_iter} iterations")
