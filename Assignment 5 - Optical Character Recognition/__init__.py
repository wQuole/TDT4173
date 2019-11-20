import os
import preprocessing
from detection import CharacterDetector
from sklearn.metrics import accuracy_score
from classification import CharacterClassifier
from sklearn.model_selection import train_test_split

# GLOBALS
CHARS74K_LITE = os.path.abspath('./dataset/chars74k-lite')
DETECTION_IMAGES = os.path.abspath('./dataset/detection-images/')
MODEL_PATH = os.path.abspath('./models/model.pkl')
PCA_PATH = os.path.abspath('./models/pca.pkl')
DETECTION_1 = 'detection-1.jpg'
DETECTION_2 = 'detection-2.jpg'

RANDOM_STATE = 42
PROBABILITY = False  # Set to True when building model for Detection


def _dataloader(filePath):
    images, classes = preprocessing.load_images(filePath)
    X, y, X_df, y_df = preprocessing.create_dataframe_and_numpy_arrays(images, classes)
    X, pca = preprocessing.pca_transform(X, 40)
    return pca, train_test_split(X, y, test_size=0.2, shuffle=True, random_state=RANDOM_STATE)


if __name__ == '__main__':
    '''
    Load and prep' data
    '''
    pca, (X_train, X_test, y_train, y_test) = _dataloader(CHARS74K_LITE)

    '''
    Classifier
    '''
    clf = CharacterClassifier(RANDOM_STATE, PROBABILITY)
    # model, predictions = clf.mlp_classifier(X_train, X_test, y_train, y_test)  # NB! Uncomment to build model
    # print(predictions)

    '''
    Save and load models (Toggle comment for usage of saving/loading your model)
    '''
    # clf.save_model(model, MODEL_PATH)
    # clf.save_pca(pca, PCA_PATH)
    model = clf.load_model(MODEL_PATH)
    pca = clf.load_pca(PCA_PATH)

    '''
    Detection (Uncomment below to run)
    '''
    d_img = preprocessing.load_single_image(DETECTION_IMAGES, DETECTION_1)
    window_size = (20,20)
    detector = CharacterDetector(d_img, model, pca, window_size)
    boundingBoxes = detector.sliding_window(stride=1)
    detector.plot_bounding_boxes(boundingBoxes)

    '''
    Test classifier
    '''
    # i = 0 # random sample index
    # sample, label = X_test[i], y_test[i]
    #
    # pred = model.predict([sample])
    # print(accuracy_score([label], pred))



