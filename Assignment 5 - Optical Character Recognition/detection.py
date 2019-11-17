from classification import CharacterClassifier


class CharacterDetector:
    def __init__(self, image, width, height):
        self.img = image
        self.w = width
        self.h = height

    def sliding_window(self, stride, scale):
        viewport = []
        for y in range(0, self.img.shape[0], stride):
            for x in range(0, self.img.shape[1], stride):
                viewport.append(self.img[y:y + scale[1], x:x + scale[0]])
        return viewport

    def pyramid(self):
        return None

    def plot_predict(self, ):
        return None