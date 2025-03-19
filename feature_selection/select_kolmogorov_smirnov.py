import numpy as np
import tqdm
from scipy import stats
from feature_selection.select_ttest import find_marker_genes
import data_loader
import pandas as pd


def select_features(k=None):
    healthy, sick, feature_names = data_loader.load_np_data()
    X = np.vstack((healthy, sick))
    y = np.hstack((np.zeros(healthy.shape[0]), np.ones(sick.shape[0])))
    # print(len(feature_names))
    pvalues = []
    for i, feature_name in tqdm.tqdm(enumerate(feature_names), total=len(feature_names)):
        cur_pvalue = stats.kstest(healthy[:, i], sick[:, i]).pvalue
        pvalues.append((i, feature_name, cur_pvalue))

    pvalues.sort(key=lambda x: x[2])
    ret = [f[1] for f in pvalues]
    if k is not None:
        return ret[:k]
    return ret

    # return feature_names[indices], indices


if __name__ == '__main__':
    print(select_features(k=1000))
    genes1 = find_marker_genes()

    print(genes1)
    genes2 = select_features()

    set_1 = set(genes1)
    set_2 = set(genes2)

    # Find intersection (common elements)
    intersection = set_1 & set_2

    # Find unique elements in each list
    unique_to_list_1 = set_1 - set_2
    unique_to_list_2 = set_2 - set_1

    # Create DataFrames from the results
    intersection_df = pd.DataFrame(intersection, columns=['Gene'])
    unique_to_list_1_df = pd.DataFrame(unique_to_list_1, columns=['Gene'])
    unique_to_list_2_df = pd.DataFrame(unique_to_list_2, columns=['Gene'])

    # Save them to CSV files
    # intersection_df.to_csv('data_cbio/intersection_genes.csv', index=False)
    # unique_to_list_1_df.to_csv('data_cbio/unique_to_list_1_genes.csv', index=False)
    # unique_to_list_2_df.to_csv('data_cbio/unique_to_list_2_genes.csv', index=False)


