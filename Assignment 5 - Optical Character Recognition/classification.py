from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
#import joblib
import pickle


class CharacterClassifier:
    def __init__(self, STATE, PROBABILITY):
        self.RANDOM_STATE = STATE
        self.proba = PROBABILITY

    def _run(self, model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)

        if self.proba:
            return model.predict_proba(X_test)
        else:
            predictions = model.predict(X_test)

        return accuracy_score(y_test, predictions) * 100

    def predict_img(self, model, image): # image must be a list
        return model.predict_proba(image)

    def svm_classifier(self,X_train, X_test, y_train, y_test):
        model = svm.SVC(kernel="linear", probability=True, gamma=5.383, C=2.67)
        return self._run(model, X_train, X_test, y_train, y_test)

    def knn_classifier(self, X_train, X_test, y_train, y_test):
        model = KNeighborsClassifier(1)
        return self._run(model, X_train, X_test, y_train, y_test)

    def mlp_classifier(self, X_train, X_test, y_train, y_test):
        model = MLPClassifier(hidden_layer_sizes=(860,), random_state=self.RANDOM_STATE, max_iter=500)
        return model, self._run(model, X_train, X_test, y_train, y_test)

    def random_forest_classifier(self, X_train, X_test, y_train, y_test):
        model = RandomForestClassifier(n_estimators=200, max_depth=100, random_state=self.RANDOM_STATE)
        return self._run(model, X_train, X_test, y_train, y_test)

    def save_model(self, model):
        print(f"saving model --> {model}")
        with open("/Users/wquole/PycharmProjects/TDT4173/Assignment 5 - Optical Character Recognition/models/model.pkl", 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, filePath):
        with open(filePath, 'rb') as f:
            model = pickle.load(f)
            print(f"loading model --> {model}")
            return model


    def save_pca(self, pca):
        print(f"saving PCA --> {pca}")
        with open("/Users/wquole/PycharmProjects/TDT4173/Assignment 5 - Optical Character Recognition/models/pca.pkl", 'wb') as f:
            pickle.dump(pca, f)

    def load_pca(self, filePath):
        with open(filePath, 'rb') as f:
            loaded = pickle.load(f)
            print(f"loading PCA -->")
            return loaded