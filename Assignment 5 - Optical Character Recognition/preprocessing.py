import os
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.feature import hog
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_images(filePath):
    images = []
    classes = []
    for root, dirs, files in sorted(os.walk(filePath)):
        for file in files:
            if file.endswith(".jpg"):
                img = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.GaussianBlur(img, (3, 3), 0)
                    #_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    img = img / 255  # normalize pixel values
                    images.append(img.ravel())  # ravel (20, 20) to (400, )
                    classes.append(file[:1])  # append classification    return images, classes
    return images, classes

def load_single_image(filePath, fileName):
    for root, dirs, files in sorted(os.walk(filePath)):
        for file in files:
            if file == fileName:
                img = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = img / 255 # normalize pixel values
                    img = np.asarray(img)
    return img


def create_dataframe_and_numpy_arrays(images, classes):
    data = np.asarray(images)
    classes = np.asarray(classes)

    df = pd.DataFrame(data=data)
    df["class"] = classes
    X = df.iloc[:, 0:df.shape[1]-1]
    y = df["class"]

    return X.values, y.values, X, y


def standardscaler_transform(X):
    return StandardScaler().fit_transform(X)


def pca_transform(X, n_c):
    pca = PCA(n_c)
    reduced = pca.fit_transform(X)
    return reduced, pca


def edge_detection(X):
    feature_vector = []
    theta_vector = []
    K_x = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1], np.float32) # Horisontal kernel/filter
    K_y = np.array([-1, -2, -1, 0, 0, 0, 1, 2, 1], np.float32) # VerticaAl kernel/filter
    # K_x = cv2.Sobel(np.float32(src_image), cv2.CV_32F, 1, 0)
    # K_y = cv2.Sobel(np.float32(src_image), cv2.CV_32F, 0, 1)

    for im in X:
        im = resize(im.reshape((20, 20)), (128,128), anti_aliasing=True, mode='reflect')
        grad_x = ndimage.filters.convolve(im.ravel(), K_x)
        grad_y = ndimage.filters.convolve(im.ravel(), K_y)

        G = np.hypot(grad_x, grad_y)
        G = G / G.max() * 255
        theta = np.arctan(grad_y, grad_x)
        feature_vector.append(G)
        theta_vector.append(theta)

    return feature_vector# Gradient and edge direction matrices

def hog_descriptor(X):
    feature_vector = []
    for im in X:
        im = im.reshape((20, 20)) # need to resize picture in order to fit 8x8 pixel cells 8 times
        f, _ = hog(im, orientations=8, pixels_per_cell=(8,8), cells_per_block=(1,1), visualize=True, multichannel=False, feature_vector=True, block_norm='L2') # orientations = number of feature buckets
        feature_vector.append(f)
    return feature_vector


def edge_detection_transform(X_train, X_test):
    _, X_train = edge_detection(X_train)
    _, X_test = edge_detection(X_test)

    return X_train, X_test


def hog_transform(X_train, X_test):
    X_train = hog_descriptor(X_train)
    X_test = hog_descriptor(X_test)

    return X_train, X_test

