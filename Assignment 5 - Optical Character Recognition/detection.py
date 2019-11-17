from classification import CharacterClassifier
import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2 import dnn

dnn.NMSBoxes()

class CharacterDetector:
    def __init__(self, image, clf, pca, scale):
        self.img = image
        self.clf = clf
        self.pca = pca
        self.scale = scale


    def check_lit_window(self, window):
        "Returns True if all elements evaluate to True" #https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.all.html#numpy.ndarray.all
        return np.ndarray.all(window)


    def classify_window(self, window): # tar inn et charClassifier object med proba satt til True
            window = window.reshape(1,-1)
            downscaled = self.pca.transform(window)
            w_prediction = self.clf.predict_proba(downscaled)
            pred_index = w_prediction.argmax()
            pred_value = w_prediction[0][pred_index]
            if pred_value >= 0.99999 and self.check_lit_window(window):
                print(f"{pred_value} --> {pred_index}")
                return pred_index, pred_value
            return -1

    def sliding_window(self, stride):
        viewports = {}
        counter = 0
        for y in range(0, self.img.shape[0] - self.scale[0], stride):
            for x in range(0, self.img.shape[1] - self.scale[0], stride):
                window = self.img[y:y + self.scale[1], x:x + self.scale[0]]
                preds = self.classify_window(window)
                if preds != -1:
                    viewports[counter] = [(y,x), preds, window]
                    counter += 1
        return viewports

    def fetch_data(self, viewports):
        predicted_char_indices = []
        predicted_char_probabilities = []
        coords = []
        for values in viewports.values():
            coords.append(values[0])
            predicted_char_indices.append(values[1][0])
            predicted_char_probabilities.append(values[1][1])

        return predicted_char_indices, predicted_char_probabilities, coords

    def plot_viewport(self, viewPort):
        _, _, coordss = self.fetch_data(viewPort)

        for coords in coordss:
            cv2.rectangle(self.img, coords, (coords[0] + self.scale[0], coords[1] + self.scale[0]), (0, 255, 0), 1)  # draw rectangle on image
            plt.imshow(self.img)
            plt.show()

        return None