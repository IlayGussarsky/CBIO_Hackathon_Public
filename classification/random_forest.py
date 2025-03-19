import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import globals
from data_loader import load_data
from sklearn.metrics import confusion_matrix

class RandomForestModel:
    def __init__(self, genes=None, n_estimators=100, max_depth=None, min_samples_split=2):
        self.genes = genes
        self.classifier = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
        )
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

    def feature_importances(self):
        return self.classifier.named_steps['randomforestclassifier'].feature_importances_

def rf_metrics():
    k = 100
    genes = globals.gene_order_fnn[:k]

    # Load globals
    healthy, sick, feature_names = globals.healthy, globals.sick, globals.feature_names

    # Split train-test
    healthy_train, healthy_test = train_test_split(healthy, test_size=0.2)
    sick_train, sick_test = train_test_split(sick, test_size=0.2)

    # Train the Random Forest model
    rf_model = RandomForestModel(genes=genes)
    rf_model.fit(healthy_train, sick_train)

    # Make predictions
    healthy_predictions = rf_model.predict(healthy_test)
    sick_predictions = rf_model.predict(sick_test)

    # Calculate metrics
    y_true = np.vstack((np.zeros((len(healthy_predictions), 1)), np.ones((len(sick_predictions), 1))))
    y_scores = np.hstack((healthy_predictions, sick_predictions))
    accuracy = accuracy_score(y_true, y_scores)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_true, y_scores))

    # Confusion matrix
    print(confusion_matrix(y_true, y_scores))

    # Feature importances
    feature_importances = rf_model.feature_importances()
    fi_df = pd.DataFrame({
        'feature': globals.feature_names[rf_model.indices],
        'importance': feature_importances
    }).sort_values('importance', ascending=False)

    print("\n=== Top 10 most important features ===")
    print(fi_df.head(10))

if __name__ == '__main__':
    rf_metrics()
