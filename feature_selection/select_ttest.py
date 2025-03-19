from scipy.stats import ttest_ind
from data_loader import load_np_data, load_data
import pandas as pd
import numpy as np


def find_marker_genes(threshold=0.001, top_n=None):
    """
    The function recives the genes matrix and find sequences in which the difference between
    helathy and sick is significant -> use t-test on those two groups and outputs
    a significance value for each sequence. Then fikter the sequences based on a threshold.
    :return: sequences who passed the threshold
    """
    healthy, sick = load_data()
    assert list(healthy.columns) == list(sick.columns), "Healthy and sick groups must have the same genes."
    healthy = healthy.rename(columns=healthy.iloc[0]).iloc[1:].reset_index(drop=True).astype(float)
    sick = sick.rename(columns=sick.iloc[0]).iloc[1:].reset_index(drop=True).astype(float)

    significant_genes = []
    p_values = {}

    for gene in healthy.columns:
        # Perform paired t-test
        t_stat, p_value = ttest_ind(healthy[gene], sick[gene], equal_var=False)

        # Modify p-value for one-sided test
        # if alternative == 'greater':
        #     # Hypothesis: Healthy mean > Sick mean (one-sided)
        #     if np.mean(healthy[gene]) > np.mean(sick[gene]):
        #         p_value /= 2  # Halve the p-value for the one-sided test (greater)
        # elif alternative == 'less':
        #     # Hypothesis: Sick mean > Healthy mean (one-sided)
        #     if np.mean(sick[gene]) > np.mean(healthy[gene]):
        #         p_value /= 2  # Halve the p-value for the one-sided test (less)

        # Check if the p-value is below the threshold
        p_values[gene] = p_value

    p_values_df = pd.DataFrame(list(p_values.items()), columns=["Gene", "p_value"])

    # Sort by p-value in ascending order
    p_values_df = p_values_df.sort_values(by="p_value")

    # Filter the top `top_n` genes with the smallest p-values
    if top_n is None:
        top_genes = p_values_df
    else:
        top_genes = p_values_df.head(top_n)
    genes1 = top_genes['Gene'].to_list()
    return genes1


if __name__ == '__main__':
    print(find_marker_genes())
