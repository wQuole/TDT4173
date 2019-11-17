from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

class CharacterClassifier:
    def __init__(self, STATE):
        self.RANDOM_SEED = STATE


    def svm_classifier(self,X_train, X_test, y_train, y_test):
        model = svm.SVC(kernel="linear", probability=True, gamma=0.001)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        score = accuracy_score(y_test, predictions)

        return score*100

    def knn_classifier(self, X_train, X_test, y_train, y_test):
        model = KNeighborsClassifier(1)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        score = accuracy_score(y_test, predictions)

        return score*100

    def mlp_classifier(self, X_train, X_test, y_train, y_test):
        model = MLPClassifier(hidden_layer_sizes=(860,), random_state=self.RANDOM_SEED, max_iter=500)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        score = accuracy_score(y_test, predictions)

        return score*100



