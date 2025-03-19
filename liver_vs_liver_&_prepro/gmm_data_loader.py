import pandas as pd

from config import HEALTHY_PATH, SICK_PATH, TEST_PATH, TRAIN_PATH


def load_data():
    """
    Load data from csv files.
    Make sure config exists and has HEALTHY_PATH and SICK_PATH defined.
    NOTE: First row is the feature names.
    :return: healthy[(samples+1) x features], sick[(samples+1) x features]
    """
    healthy = pd.read_csv(HEALTHY_PATH).T
    sick = pd.read_csv(SICK_PATH).T
    return healthy, sick


def load_np_data():
    """
    Loads data from csv files and converts them to numpy arrays.

    :return: healthy[samples x features], sick[samples x features], feature_names[features]
    """
    healthy, sick = load_data()

    # To numpy
    healthy = healthy.values
    sick = sick.values

    assert healthy.shape[1] == sick.shape[1], 'Different number of features'
    assert all([healthy[0][i] == sick[0][i] for i in range(healthy.shape[1])]), 'Different features'

    # Feature number to feature name
    feature_names = healthy[0]

    # Remove first row (feature names)
    healthy = healthy[1:]
    sick = sick[1:]
    return healthy, sick, feature_names


def load_merged_data():
    """
    Load data from csv files and merge them into one dataset.
    Make sure config exists and has HEALTHY_PATH and SICK_PATH defined.
    NOTE: First row is the feature names.
    :return: data[(samples+1) x features]
    """

    train = pd.read_csv(TRAIN_PATH)
    train = train.drop(columns=[train.columns[0]]).to_numpy()[:, -26:]
    train_healthy = train[train[:, -1] == 0][:, :-1]
    train_sick = train[train[:, -1] == 1][:, :-1]

    test = pd.read_csv(TEST_PATH)
    test = test.drop(columns=[test.columns[0]]).to_numpy()[:, -26:]
    test_healthy = test[test[:, -1] == 0][:, :-1]
    test_sick = test[test[:, -1] == 1][:, :-1]

    return train_healthy, train_sick, test_healthy, test_sick

