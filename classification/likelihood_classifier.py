import numpy as np

from classification.likelihood_calculator import LikelihoodCalculator


class LikelihoodClassifier:
    def __init__(self, Xs: list[np.ndarray], type: str):
        """
        :param Xs: list of datasets, each is [samples x features]
        :param type: how to calculate the likelihood. Options: "Gaussian", "KDE"
        """
        assert type in ["Gaussian", "KDE"], "Invalid type"
        self.likelihood_estimators = [LikelihoodCalculator(X, type) for X in Xs]

    def classify(self, data):
        """
        Does classification.
        :param data: data_cbio to classify [samples x features]
        :return: np.array, the index of the dataset that has the highest likelihood for each row.
        """
        return np.asarray([[l.calculate_single_datapoint(row) for l in self.likelihood_estimators] for row in data])


if __name__ == '__main__':
    Xs = [np.asarray([[1, 1], [1.1, 1.1]]), np.asarray([[2, 2], [2.1, 2.1]])]
    for row in Xs[0]:
        print(row)
    classifier = LikelihoodClassifier(Xs, "Gaussian")
    print(classifier.classify(np.asarray([[0.5, 2.5]])))
