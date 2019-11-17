from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


class CharacterClassifier:
    def __init__(self, STATE):
        self.RANDOM_STATE = STATE

    def _run(self, model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        score = accuracy_score(y_test, predictions)

        return score * 100

    def svm_classifier(self,X_train, X_test, y_train, y_test):
        model = svm.SVC(kernel="linear", probability=True, gamma=5.383, C=2.67)
        return self._run(model, X_train, X_test, y_train, y_test)

    def knn_classifier(self, X_train, X_test, y_train, y_test):
        model = KNeighborsClassifier(1)
        return self._run(model, X_train, X_test, y_train, y_test)

    def mlp_classifier(self, X_train, X_test, y_train, y_test):
        model = MLPClassifier(hidden_layer_sizes=(860,), random_state=self.RANDOM_STATE, max_iter=500)
        return self._run(model, X_train, X_test, y_train, y_test)

    def random_forest_classifier(self, X_train, X_test, y_train, y_test):
        model = RandomForestClassifier(n_estimators=200, max_depth=100, random_state=self.RANDOM_STATE)
        return self._run(model, X_train, X_test, y_train, y_test)
