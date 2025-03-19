import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import globals
from data_loader import load_data
from sklearn.metrics import confusion_matrix


class KNNClassifier:
    def __init__(self, genes: list[str] = None, k=5):
        self.genes = genes
        self.classifier = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
        self.indices = [np.where(globals.feature_names == g)[0][0] for g in genes]


    def fit(self, healthy, sick):
        feature_names = globals.feature_names
        genes = self.genes
        if self.genes is None:
            genes = feature_names
        # Get indices of genes in feature_names
        healthy = healthy[:, self.indices]
        sick = sick[:, self.indices]

        X = np.vstack((healthy, sick))
        y = np.hstack((np.zeros(len(healthy)), np.ones(len(sick))))
        self.classifier.fit(X, y)

    def predict(self, data) -> np.ndarray:
        data = data[:, self.indices]
        return self.classifier.predict(data)


def knn_metrics():
    k = 100
    genes = globals.gene_order_fnn[:k]

    # Load globals
    healthy, sick, feature_names = globals.healthy, globals.sick, globals.feature_names

    # Split train-test
    healthy_train, healthy_test = train_test_split(healthy, test_size=0.2)
    sick_train, sick_test = train_test_split(sick, test_size=0.2)

    # Train the k-NN model
    knn_classifier = KNNClassifier(genes=genes)
    knn_classifier.fit(healthy_train, sick_train)

    # Make predictions
    healthy_predictions = knn_classifier.predict(healthy_test)
    sick_predictions = knn_classifier.predict(sick_test)

    # Calculate metrics
    y_true = np.vstack((np.zeros((len(healthy_predictions), 1)), np.ones((len(sick_predictions), 1))))
    y_scores = np.hstack((healthy_predictions, sick_predictions))
    accuracy = accuracy_score(y_true, y_scores)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_true, y_scores))

    # Confusion matrix
    print(confusion_matrix(y_true, y_scores))

if __name__ == '__main__':
    knn_metrics()
