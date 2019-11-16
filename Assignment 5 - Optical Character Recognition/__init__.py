import os
import numpy as np
from classification import CharacterClassifier
import preprocessing
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

if __name__ == '__main__':
    # Load and prep' data
    chars74k_lite = os.path.abspath('/Users/wquole/PycharmProjects/TDT4173/Assignment 5 - Optical Character Recognition/dataset/chars74k-lite')
    images, classes = preprocessing.load_data(chars74k_lite)
    X, y, X_array, y_array = preprocessing.create_dataframe_and_numpy_arrays(images, classes)
    # Split data 80/20
    X_array = preprocessing.standarscaler_transform(X_array)
    X_array = preprocessing.pca_transform(X_array, 40)
    X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size=0.2, shuffle=True, random_state=RANDOM_STATE)
    X_train, X_test = preprocessing.sobel_transform(X_train, X_test)
    # Classifier
    clf = CharacterClassifier(RANDOM_STATE)
    predictions = clf.svm_classifier(X_train, X_test, y_train, y_test)
    print(predictions)


'''
TODO:
checkout out:
https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
https://www.youtube.com/watch?v=NfiCmhLLxMA
https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html#sklearn.pipeline.make_pipeline
https://en.wikipedia.org/wiki/Eigenface
https://www.algosome.com/articles/optical-character-recognition-java.html
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