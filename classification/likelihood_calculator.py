import numpy as np
from scipy.stats import gaussian_kde, norm


class LikelihoodCalculator:
    def __init__(self, X: np.ndarray, type: str):
        """

        :param X: the data_cbio [samples x features]
        :param type: how to calculate the likelihood. Options: "Gaussian", "KDE"
        """
        assert type in ["Gaussian", "KDE"], "Invalid type"
        self.X = X
        self.type = type
        if type == "Gaussian":
            self.means = [np.mean(X[:, i]) for i in range(X.shape[1])]
            self.stds = [np.std(X[:, i]) for i in range(X.shape[1])]
            self.log_densities = [lambda x: norm.logpdf(x, self.means[i], self.stds[i]) for i in range(X.shape[1])]

        if type == "KDE":
            self.densities = [gaussian_kde(X[:, i], bw_method='scott') for i in range(X.shape[1])]

    def calculate_single_datapoint(self, data):
        if self.type == "Gaussian":
            ret = sum([d(data[i]) for i, d in enumerate(self.log_densities)])
            # print(ret)
            return ret
        return sum([np.log(d(data[i])) for i, d in enumerate(self.densities)])

    def calculate(self, data):
        return [self.calculate_single_datapoint(row) for row in data]

