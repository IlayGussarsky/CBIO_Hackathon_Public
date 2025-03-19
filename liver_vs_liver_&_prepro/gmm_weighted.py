import numpy as np
from scipy.stats import norm
import gmm_data_loader  # Assuming this is your module for loading data
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance

class GeneClassifier:
    def __init__(self, gene_data_health, gene_data_sick):
        # Fit the normal distribution for both healthy and sick groups
        self.mean_health = np.mean(gene_data_health)
        self.std_health = np.std(gene_data_health)
        self.mean_sick = np.mean(gene_data_sick)
        self.std_sick = np.std(gene_data_sick)

    def calculate_probabilities(self, x):
        # Calculate the probability densities for health and sick
        prob_health = norm.pdf(x, self.mean_health, self.std_health)
        prob_sick = norm.pdf(x, self.mean_sick, self.std_sick)
        return prob_health, prob_sick

    def classify(self, x):
        # Get probabilities for being healthy and sick for this gene
        prob_health, prob_sick = self.calculate_probabilities(x)
        return prob_health, prob_sick


class MultiGeneClassifier:
    def __init__(self, gene_data_health, gene_data_sick, num_genes=None):
        self.classifiers = []
        self.weights = []

        # Use only a subset of genes if num_genes is specified
        if num_genes is None:
            num_genes = len(gene_data_health[0])  # Use all genes by default

        # Limit the number of genes to the specified number
        self.gene_data_health = gene_data_health[:, :num_genes]
        self.gene_data_sick = gene_data_sick[:, :num_genes]

    def train(self):
        """
        Train the classifier by fitting each gene to its respective healthy and sick data
        and calculate weights based on Wasserstein distance.
        """
        for gene_health, gene_sick in zip(self.gene_data_health.T, self.gene_data_sick.T):
            # Fit a GeneClassifier for the current gene
            self.classifiers.append(GeneClassifier(gene_health, gene_sick))

            # Compute Wasserstein distance between healthy and sick distributions for the gene
            distance = wasserstein_distance(gene_health, gene_sick)
            self.weights.append(distance)

        # Normalize weights to sum to 1
        total_weight = sum(self.weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]

    def predict(self, input_vector):
        """
        For each classifier (gene), calculate the probability densities for health and sick.
        """
        decision_vector = []
        for classifier, x in zip(self.classifiers, input_vector):
            prob_health, prob_sick = classifier.classify(x)
            decision_vector.append([prob_health, prob_sick])
        return decision_vector

    def final_decision(self, input_vector):
        """
        Get the final classification based on the weighted probabilities from all genes.
        """
        # Get decision vector from all classifiers (genes)
        decision_vector = self.predict(input_vector)

        # Calculate the weighted probability for being healthy and sick
        weighted_prob_health = 0.0
        weighted_prob_sick = 0.0
        for i in range(len(self.classifiers)):
            prob_health, prob_sick = decision_vector[i]
            weighted_prob_health += self.weights[i] * prob_health
            weighted_prob_sick += self.weights[i] * prob_sick

        # The final classification is based on which weighted probability is higher
        if weighted_prob_health > weighted_prob_sick:
            return 0  # Healthy
        else:
            return 1  # Sick


# Assuming the module `gmm_data_loader` has a function `load_merged_data`
train_healthy, train_sick, test_healthy, test_sick = gmm_data_loader.load_merged_data()

# Initialize the classifier with the specified number of genes
classifier = MultiGeneClassifier(train_healthy, train_sick)

# Train the classifier
classifier.train()

# Function to make predictions on the test set
def predict_all(classifier, test_data):
    predictions = []
    for sample in test_data:
        prediction = classifier.final_decision(sample)
        predictions.append(prediction)
    return np.array(predictions)

# Combine healthy and sick test sets
test_data = np.vstack((test_healthy, test_sick))
true_labels = np.array([0] * len(test_healthy) + [1] * len(test_sick))

# Get predictions on test data
predictions = predict_all(classifier, test_data)

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, predictions, normalize='true')

# Display confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", xticklabels=["Healthy", "Sick"], yticklabels=["Healthy", "Sick"])
plt.title("Confusion Matrix")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()

# Display classification report (precision, recall, f1-score)
print(classification_report(true_labels, predictions, target_names=["Healthy", "Sick"]))

from sklearn.metrics import roc_curve, auc


# Function to get probabilities for the positive class (sick) from the classifier
def predict_probabilities(classifier, test_data):
    probabilities = []
    for sample in test_data:
        # Get the probability vector for health and sick for the sample
        decision_vector = classifier.predict(sample)
        # Sum weighted probabilities for the sick class
        weighted_prob_sick = sum([classifier.weights[i] * prob_sick
                                  for i, (_, prob_sick) in enumerate(decision_vector)])
        probabilities.append(weighted_prob_sick)
    return np.array(probabilities)


# Get probabilities for the positive class (sick)
probabilities = predict_probabilities(classifier, test_data)

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Random guessing line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()
