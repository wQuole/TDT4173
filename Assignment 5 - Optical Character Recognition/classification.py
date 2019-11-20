import pickle
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


class CharacterClassifier:
    def __init__(self, STATE, PROBABILITY):
        self.RANDOM_STATE = STATE
        self.proba = PROBABILITY

    def _run(self, model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        if self.proba:
            predictions = model.predict_proba(X_test)
            return model, predictions
        else:
            predictions = model.predict(X_test)
            return model, accuracy_score(y_test, predictions) * 100

    def svm_classifier(self,X_train, X_test, y_train, y_test):
        model = svm.SVC(kernel="linear", probability=True, gamma=5.383, C=2.67)
        return self._run(model, X_train, X_test, y_train, y_test)

    def knn_classifier(self, X_train, X_test, y_train, y_test):
        model = KNeighborsClassifier(1)
        return self._run(model, X_train, X_test, y_train, y_test)

    def mlp_classifier(self, X_train, X_test, y_train, y_test):
        model = MLPClassifier(hidden_layer_sizes=(850,), random_state=self.RANDOM_STATE, max_iter=1000)
        return self._run(model, X_train, X_test, y_train, y_test)

    def random_forest_classifier(self, X_train, X_test, y_train, y_test):
        model = RandomForestClassifier(n_estimators=200, max_depth=200, random_state=self.RANDOM_STATE)
        return self._run(model, X_train, X_test, y_train, y_test)

    def save_model(self, model, filePath):
        print(f"Saving model:\n{model}")
        with open(filePath, 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, filePath):
        with open(filePath, 'rb') as f:
            model = pickle.load(f)
            print(f"Loading model ...")
            return model

    def save_pca(self, pca, filePath):
        print(f"Saving PCA:\n{pca}")
        with open(filePath, 'wb') as f:
            pickle.dump(pca, f)

    def load_pca(self, filePath):
        with open(filePath, 'rb') as f:
            loaded = pickle.load(f)
            print("Loading PCA ...")
            return loaded