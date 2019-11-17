from classification import CharacterClassifier
import cv2
import matplotlib.pyplot as plt
import numpy as np


class CharacterDetector:
    def __init__(self, image, width, height, clf):
        self.img = image
        self.w = width
        self.h = height
        self.clf = clf
        self.hits = []

    def classify_window(self, window): # tar inn et charClassifier object med proba satt til True
        w_prediction = self.clf.predict_proba(window)
        print('yo')
        if w_prediction[0] > 0.69:
            self.hits.append((window, w_prediction))
            return True
        return False

    def sliding_window(self, stride, scale):
        viewport = []
        for y in range(0, self.img.shape[0], stride):
            for x in range(0, self.img.shape[1], stride):
                viewport.append(self.img[y:y + scale[1], x:x + scale[0]])
                window = self.img[y:y + scale[1], x:x + scale[0]]
                tmp = self.img
                if self.classify_window(window):
                #f x > 30 and x < 100 and y > 20:
                    print('character found!')
                    cv2.rectangle(tmp, (x, y), (x + self.w, y + self.h), (0, 255, 0), 1)  # draw rectangle on image
                    plt.imshow(np.array(tmp).astype('uint8'))

                plt.show()

        return viewport

    def pyramid(self):
        return None

    def plot_viewport(self,):
        pass

        return None