import cv2
import matplotlib.pyplot as plt
import math
from string import ascii_lowercase
import numpy as np


class CharacterDetector:
    def __init__(self, image, clf, pca, scale):
        self.img = image
        self.clf = clf
        self.pca = pca
        self.scale = scale[0]

    def euclidean_distance(self,c1, c2):
        return math.sqrt(sum([(c1[0] - c2[0])**2, (c1[1] - c2[1])**2]))

    def check_corners_of_window(self, X_COORD, Y_COORD):
        y,x  = X_COORD, Y_COORD
        upper_left = self.img[x][y]
        upper_right = self.img[x][y+self.scale]
        lower_left =  self.img[x+self.scale][y]
        lower_right = self.img[x+self.scale][y+self.scale]
        if (upper_right+upper_left+lower_left+lower_right) >= 3.9:
            return False
        return True

    def classify_box(self, window):
            window = window.flatten()
            downscaled = self.pca.transform([window])
            w_prediction = self.clf.predict_proba(downscaled)
            pred_index = w_prediction.argmax()
            pred_value = w_prediction[0][pred_index]
            if pred_value >= 0.99999995:
                return pred_index, pred_value
            return -1

    def sliding_window(self, stride):
        boxIndex = 0
        boundingBoxes = {}
        previousBoundingBoxCoords = (0,0)
        threshold = 19
        first = True
        for y in range(0, self.img.shape[0] - self.scale, stride):
            for x in range(0, self.img.shape[1] - self.scale, stride):
                box = self.img[y:y + self.scale, x:x + self.scale]
                predictions = self.classify_box(box)
                if predictions != -1:
                    distance = self.euclidean_distance((x, y), previousBoundingBoxCoords)
                    if first or distance >= threshold:
                        first = False
                        boundingBoxes[boxIndex] = [(x,y), predictions, box]
                        previousBoundingBoxCoords = (x,y)
                        boxIndex += 1
        return boundingBoxes

    def plot_bounding_boxes(self, boundingBoxes):
        for prediction in boundingBoxes.values():
            startX = prediction[0][0]
            startY = prediction[0][1]
            endX = prediction[0][0] + self.scale
            endY = prediction[0][1] + self.scale
            char = ascii_lowercase[prediction[1][0]]

            cv2.rectangle(self.img, (startX, startY), (endX, endY), (0, 255, 0), 1)  # draw rectangle on image
            cv2.putText(self.img, char, (startX, startY + 2 * self.scale), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            cv2.
            plt.imshow(self.img, cmap="gray", interpolation='nearest')
        plt.show()

    def plot_nms_boxes(self, boundingBoxes):
        print(f"Initial amount of boxes:\t{len(boundingBoxes)}")
        choices =  self.non_max_suppression(boundingBoxes, 1)
        print(f"After NMS:\t{len(choices)}")
        for (startX, startY, endX, endY) in choices:
            cv2.rectangle(self.img, (startX, startY), (startX+self.scale, startY+self.scale), (0, 255, 0), 1)
            plt.imshow(self.img, cmap='gray', interpolation='nearest')
        plt.show()

    def fetch_coords(self, boundingBoxes):
        coords = []
        for predictions in boundingBoxes.values():
            startX = predictions[0][0]
            startY = predictions[0][1]
            endX = predictions[0][0] + self.scale
            endY = predictions[0][1] + self.scale

            coords.append((startX, startY, endX, endY))
        return coords

    def non_max_suppression(self, boundingBoxes, threshold):
        if len(boundingBoxes) == 0:
            return []

        choices = []
        x1, y1 = boundingBoxes[:, 0], boundingBoxes[:, 1]
        x2, y2 = boundingBoxes[:, 2], boundingBoxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1) # Add 1 to avoid multiplying with since coords is 0-initialized
        indices = np.argsort(y2)

        while len(indices) > 0:
            lastIndex = len(indices) - 1
            i = indices[lastIndex]
            choices.append(i)
            suppress = [lastIndex]

            for pos in range(0, lastIndex): # loop over all indices
                j = indices[pos] # current index

                x1_max, y1_max = max(x1[i], x1[j]), max(y1[i], y1[j])
                x2_max, y2_max = max(x2[i], x2[j]), max(y2[i], y2[j])

                width = max(0, (x2_max - x1_max) + 1)
                height = max(0, (y2_max - y1_max) + 1)

                overlappingArea = float(width * height)/area[j]

                if overlappingArea > threshold:
                    print(pos, x1_max, y1_max, x2_max, y2_max)
                    suppress.append(pos)

            indices = np.delete(indices, suppress) # update indices by removing suppressed indices

        return boundingBoxes[choices]





