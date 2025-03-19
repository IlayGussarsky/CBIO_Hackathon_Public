import numpy as np
import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# Set numpy seed
np.random.seed(42)
from sklearn.metrics import roc_curve, roc_auc_score
import globals


class MeanClassifier:
    def __init__(self, genes: list[str] = None):
        self.genes = genes
        # self.ensemble = []
        self.lines = []
        self.means = []

    def fit(self, healthy, sick):
        feature_names = globals.feature_names
        genes = self.genes
        if self.genes is None:
            genes = feature_names
        for g in genes:
            index = np.where(feature_names == g)[0][0]
            healthy_mean = np.mean(healthy[:, index])
            sick_mean = np.mean(sick[:, index])
            self.lines.append((healthy_mean + sick_mean) / 2)
            self.means.append((healthy_mean, sick_mean))

    def predict_single_sample(self, single_sample):
        votes = []
        for i, g in enumerate(self.genes):
            index = np.where(globals.feature_names == g)[0][0]
            if self.means[i][0] > self.means[i][1]:
                votes.append(1 if single_sample[index] < self.lines[i] else 0)
            else:
                votes.append(0 if single_sample[index] < self.lines[i] else 1)

        # Majority vote decision
        return 1 if sum(votes) > len(self.genes) / 2 else 0

    def predict(self, data) -> np.ndarray:
        return np.asarray([self.predict_single_sample(row) for row in tqdm.tqdm(data)])


def plot_roc_curve(GENES, label="ROC curve"):
    healthy, sick, feature_names = globals.healthy, globals.sick, globals.feature_names
    # Split to train, test
    healthy_train, healthy_test = train_test_split(healthy, test_size=0.2)
    sick_train, sick_test = train_test_split(sick, test_size=0.2)
    classifier = MeanClassifier(genes=GENES)
    classifier.fit(healthy_train, sick_train, )
    healthy_predictions = classifier.predict(healthy_test)
    sick_predictions = classifier.predict(sick_test)

    # Draw ROC curve
    y_true = np.vstack((np.zeros((len(healthy_predictions), 1)), np.ones((len(sick_predictions), 1))))
    y_scores = np.hstack((healthy_predictions, sick_predictions))
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    # plt.figure()
    plt.plot(fpr, tpr, label=label)
    # for i, threshold in enumerate(thresholds):
    #     plt.annotate(f'{threshold:.2f}', (fpr[i], tpr[i]), textcoords="offset points", xytext=(0, 10), ha='center')


if __name__ == '__main__':
    k = 500
    num_features = [30, 100, 500, 200, 50]
    for l in num_features:
        plt.figure()
        GENES = globals.gene_order_ttest[:l]
        plot_roc_curve(GENES, label='ttest')
        GENES = globals.gene_order_kstest[:l]
        plot_roc_curve(GENES, label='kstest')
        GENES = globals.gene_order_gmm[:l]
        plot_roc_curve(GENES, label='gmm')
        GENES = globals.gene_order_fnn[:l]
        plot_roc_curve(GENES, label='fnn')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC curve {l} features')
        plt.legend()
        plt.grid(True)
        plt.show()
