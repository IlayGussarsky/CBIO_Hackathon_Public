from knn_classifier import KNNClassifier
from weighted_ll import MultiGeneClassifier
from mean_ensemble_classifier import MeanClassifier
from random_forest import RandomForestModel
import globals
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def plot_confusion_matrix(GENES, CLF, label="Confusion Matrix"):
    healthy, sick, feature_names = globals.healthy, globals.sick, globals.feature_names
    # Split to train, test
    healthy_train, healthy_test = train_test_split(healthy, test_size=0.2)
    sick_train, sick_test = train_test_split(sick, test_size=0.2)

    # Initialize and fit classifier
    classifier = CLF(genes=GENES)
    classifier.fit(healthy_train, sick_train)

    # Make predictions
    healthy_predictions = classifier.predict(healthy_test)
    sick_predictions = classifier.predict(sick_test)

    # Combine true labels and predictions
    y_true = np.hstack((np.zeros(len(healthy_test)), np.ones(len(sick_test))))
    y_pred = np.hstack((healthy_predictions, sick_predictions))

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize by row

    # Round the confusion matrix values
    cm_normalized = np.round(cm_normalized, 2)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=["Healthy", "Sick"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(label)
    plt.show()
    plt.close()

if __name__ == '__main__':
    # KNN
    # GENES = globals.gene_order_fnn[:500]
    # plot_confusion_matrix(GENES, CLF=KNNClassifier, label="Confusion Matrix \n KNN Classifier (FNN, 500)")
    # MeanEnsemble
    GENES = globals.gene_order_fnn[:500]
    plot_confusion_matrix(GENES, CLF=MeanClassifier, label="Confusion Matrix \n Mean classifier (FNN, 500)")
    # # Random forest - no feature selection
    GENES = globals.gene_order_fnn
    plot_confusion_matrix(GENES, CLF=KNNClassifier, label="Confusion Matrix \n KNN classifier (All)")
    # # Random forest - with feature selection
    # # Weighted Log Likelihood
    # GENES = globals.gene_order_ttest[:70]
    # plot_confusion_matrix(GENES, CLF=MultiGeneClassifier)

    # MeanEnsamble - best accuracy
    GENES = globals.gene_order_kstest[:50]
    plot_confusion_matrix(GENES, CLF=MeanClassifier, label="Confusion Matrix \n Mean classifier (KS test, 50)")

    # KNN  - best Accuracy
    GENES = globals.gene_order_ttest[:100]
    plot_confusion_matrix(GENES, CLF=KNNClassifier, label="Confusion Matrix \n KNN Classifier (t-test, 100)")

    # Random Forest - best TPR
    GENES = globals.gene_order_ttest[:60]
    plot_confusion_matrix(GENES, CLF=RandomForestModel, label="Confusion Matrix \n Random Forest Classifier (t-test, 60)")

    # KNN - best TPR
    GENES = globals.gene_order_fnn[:500]
    plot_confusion_matrix(GENES, CLF=KNNClassifier, label="Confusion Matrix \n KNN Classifier (FNN, 500)")

