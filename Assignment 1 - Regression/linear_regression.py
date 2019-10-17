from sys import argv
from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def h(x, weights):
    return x.dot(weights)


def calculateWeights(X, y):
    return np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y))


def preprocessData(filePath, xs):
    df = pd.read_csv(filePath)
    X = df[xs]
    X.insert(0, 'x0', 1, True) # Add new dimension because of merging bias into weight vector --> w0 = bias
    y = df[['y']]
    return X, y, df


def meanSquaredError(y_pred, y_actual):
    errorFrame = pd.DataFrame(columns=['y_pred', 'y_actual', 'mse'])
    errorFrame['y_pred'] = y_pred['y']
    errorFrame['y_actual'] = y_actual['y']
    errorFrame['mse'] = (y_pred['y'] - y_actual['y'])**2

    model_error = sum(errorFrame['mse'])/len(errorFrame['mse'])
    return model_error


def train(filePath, xs=['x1']):
    X_train, y_train, df_train = preprocessData(filePath, xs)

    weights = calculateWeights(X_train, y_train)

    pred_ys_train = h(X_train, weights)
    pred_ys_train = pred_ys_train.rename(columns={0: 'y'})

    return X_train, y_train, pred_ys_train, weights


def test(filePath, weights, xs=['x1']):
    X_test, y_test, df_test = preprocessData(filePath, xs)

    pred_ys_test = h(X_test, weights)
    pred_ys_test = pred_ys_test.rename(columns={0: 'y'})

    return X_test, y_test, pred_ys_test



def plotting(X, y, pred_y, title, acc, threeDim=False):
    if not threeDim:
        X = X.drop(['x0'], axis=1)
        plt.plot(X, y, 'o', label='Original data', markersize=10)
        plt.plot(X, pred_y, 'r', label='Fitted line')
        plt.title(title)
        plt.suptitle(f"Loss: {acc*100}%", va='top')
        plt.legend()
        plt.show()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        datapoints = ax.scatter(X['x1'], X['x2'], y, marker='v')
        predictions = ax.scatter(X['x1'], X['x2'], pred_y, marker='^')

        plt.title(title)
        plt.suptitle(f"Loss: {acc * 100}%", va='top')
        ax.legend([datapoints, predictions], ['datapoints', 'predictions'], numpoints=1)

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')

        plt.show()



def run(d=1):
    # DATASET
    train_1d = path.abspath("dataset/regression/train_1d_reg_data.csv")
    train_2d = path.abspath("dataset/regression/train_2d_reg_data.csv")

    test_1d = path.abspath("dataset/regression/test_1d_reg_data.csv")
    test_2d = path.abspath("dataset/regression/test_2d_reg_data.csv")

    # 1D
    if d == 1:
        # TRAINING
        X_train_1d, y_train_1d, pred_ys_train_1d, weights_1d = train(train_1d)

        print(f"bias:\t{weights_1d[0][0]}\nslope:\t{weights_1d[1][0]}")

        trained_model_error_1d = round(meanSquaredError(pred_ys_train_1d, y_train_1d), 4)
        print(f"Trained 1D-model error is: {trained_model_error_1d}"
              f"\n ~ {trained_model_error_1d * 100}%")

        plotting(X_train_1d, y_train_1d, pred_ys_train_1d, "Trained 1D", trained_model_error_1d)

        # TESTING
        X_test_1d, y_test_1d, pred_ys_test_1d = test(test_1d, weights_1d)

        tested_model_error_1d = round(meanSquaredError(pred_ys_test_1d, y_test_1d), 4)
        print(f"Tested 1D-model error is: {tested_model_error_1d}"
              f"\n ~ {tested_model_error_1d * 100}%")

        plotting(X_test_1d, y_test_1d, pred_ys_test_1d, "Tested 1D", tested_model_error_1d)


    # 2D
    elif d == 2:
        # TRAINING
        X_train_2d, y_train_2d, pred_ys_train_2d, weights_2d = train(train_2d, xs=['x1', 'x2'])
        print(f"w0 = bias:\t{weights_2d[0][0]}\nw1:\t\t{weights_2d[1][0]}\nw2:\t\t{weights_2d[2][0]}")

        trained_model_error_2d = round(meanSquaredError(pred_ys_train_2d, y_train_2d), 4)
        print(f"Trained 2D-model error is: {trained_model_error_2d}\n ~ {trained_model_error_2d * 100}%")

        plotting(X_train_2d, y_train_2d, pred_ys_train_2d, "Trained 2D", trained_model_error_2d, threeDim=1)

        # TESTING
        X_test_2d, y_test_2d, pred_ys_test_2d = test(test_2d, weights_2d, xs=['x1', 'x2'])

        tested_model_error_2d = round(meanSquaredError(pred_ys_test_2d, y_test_2d), 4)
        print(f"Tested 2D-model error is: {tested_model_error_2d}"
              f"\n ~ {tested_model_error_2d * 100}%")

        plotting(X_test_2d, y_test_2d, pred_ys_test_2d, "Tested 2D", tested_model_error_2d, threeDim=1)

    else:
        print("'1' --> 1D\n'2' --> 2D")


try:
    run(int(argv[1]))
except Exception as e:
    print("Use '1' or '2' to tell sys.argv which dataset to run")
    print(f"Exception!\n{e}")