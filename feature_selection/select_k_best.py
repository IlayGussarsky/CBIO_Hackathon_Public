import numpy as np
import tqdm
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import data_loader


def select_features(k=100):
    healthy, sick, feature_names = data_loader.load_np_data()
    X = np.vstack((healthy, sick))
    y = np.hstack((np.zeros(healthy.shape[0]), np.ones(sick.shape[0])))

    # Remove constant features
    constant_filter = VarianceThreshold(threshold=0)
    X = constant_filter.fit_transform(X)
    feature_names = feature_names[constant_filter.get_support()]

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Standardize the data_cbio
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    selector = SelectKBest(f_classif, k=k)
    selector.fit(X, y)

    # Get the indices of the selected features
    indices = selector.get_support(indices=True)
    return feature_names[indices]


if __name__ == '__main__':
    for i in select_features(k=100):
        print(i)
