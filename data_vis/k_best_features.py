import matplotlib.pyplot as plt
import numpy as np


import data_loader
from feature_selection.select_kolmogorov_smirnov import select_features
from feature_selection.select_gmm import select_gmm

def visualize_k_best(features):
    healthy, sick, feature_names = data_loader.load_np_data()
    for f in features:
        feature_index = np.where(feature_names == f)[0][0]
        plt.hist(healthy[:, feature_index], bins=np.arange(min(healthy[:, feature_index]), max(healthy[:, feature_index]) + 1, 1), edgecolor='black', density=True)
        plt.hist(sick[:, feature_index], bins=np.arange(min(sick[:, feature_index]), max(sick[:, feature_index]) + 1, 1), edgecolor='red', density=True)

        # # Add gaussians over the histograms
        # healthy_mean = np.mean(healthy[:, feature_index])
        # healthy_std = np.std(healthy[:, feature_index])
        # sick_mean = np.mean(sick[:, feature_index])
        # sick_std = np.std(sick[:, feature_index])
        # x = np.linspace(min(healthy[:, feature_index]), max(sick[:, feature_index]), 1000)
        # plt.plot(x, 1/(healthy_std * np.sqrt(2 * np.pi)) * np.exp( - (x - healthy_mean)**2 / (2 * healthy_std**2) ), color='blue')
        # plt.plot(x, 1/(sick_std * np.sqrt(2 * np.pi)) * np.exp( - (x - sick_mean)**2 / (2 * sick_std**2) ), color='green')
        plt.title(f)
        plt.xlabel(f'{f} Values')  # Title for the x-axis
        plt.ylabel('Density')
        plt.grid(True)
        plt.show()



if __name__ == '__main__':
    features = select_gmm()
    visualize_k_best(features[:2])
