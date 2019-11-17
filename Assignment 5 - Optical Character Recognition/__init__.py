import os
import preprocessing
import cv2
from classification import CharacterClassifier
from detection import CharacterDetector
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

if __name__ == '__main__':
    # Load and prep' data
    chars74k_lite = os.path.abspath('/Users/wquole/PycharmProjects/TDT4173/Assignment 5 - Optical Character Recognition/dataset/chars74k-lite')
    images, classes = preprocessing.load_data(chars74k_lite)
    X, y, X_array, y_array = preprocessing.create_dataframe_and_numpy_arrays(images, classes)
    # Split data 80/20
    X_array = preprocessing.pca_transform(X_array, 40)
    X_array = preprocessing.standarscaler_transform(X_array)
    X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size=0.2, shuffle=True, random_state=RANDOM_STATE)
    X_train, X_test = preprocessing.edge_detection_transform(X_train, X_test)
    # Classifier
    clf = CharacterClassifier(RANDOM_STATE)
    predictions = clf.mlp_classifier(X_train, X_test, y_train, y_test)
    print(predictions)
    # Detection
    detection_image = ""
    d_img = cv2.imread(detection_image, 20, 20)
    detector = CharacterDetector(d_img)
    #detector.sliding_window()



'''
TODO:
- use pickle to save and load models
- implement detection of characters

checkout out:
http://rpubs.com/Sharon_1684/454441
https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
https://www.youtube.com/watch?v=NfiCmhLLxMA
https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py
https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html#sklearn.pipeline.make_pipeline
https://scikit-learn.org/stable/modules/decomposition.html#pca
https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html
https://en.wikipedia.org/wiki/Eigenface
https://www.algosome.com/articles/optical-character-recognition-java.html
https://charlesreid1.github.io/circe/Digit%20Classification%20-%20PCA.html
https://docs.opencv.org/master/d3/d02/tutorial_py_svm_index.html
https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gacea54f142e81b6758cb6f375ce782c8d
'''


'''
 #
    # # Plotting 🔥
    # fig, (resizedImage, hogImage) = plt.subplots(1, 2, figsize=(10, 10), sharex=False, sharey=False)
    #
    # # Plot input image
    # _, inputImage = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True)
    # inputImage.imshow(img.reshape(20,20), cmap='gray') # original picture
    # inputImage.set_title('Input Image')
    #
    # resizedImage.imshow(resized_img, cmap='gray')
    # resizedImage.set_title('Resized picture')
    #
    # # Plot HOG image
    # hogImage.imshow(hog_image, 'gray')
    # hogImage.set_title('Histogram of Oriented Gradients')

    #plt.show()
    #return f_matrix #feature matrix
'''