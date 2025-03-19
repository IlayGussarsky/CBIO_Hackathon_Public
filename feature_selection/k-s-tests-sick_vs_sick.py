import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests
import tqdm

# Load gene expression matrix
gene_matrix_no_gene_transposed = pd.read_csv(
    "/Users/hilleldravish/Library/CloudStorage/GoogleDrive-darvish.hillel@mail.huji.ac.il/My Drive/CBIO-HACK/Data/CleanUp/cleaned_llb_matrix.csv"
)

# Perform K-S test for each gene comparing each diagnosis to all others
def ks_test_all_vs_rest(gene_matrix_no_gene_transposed):
    results = []
    diagnosis_levels = gene_matrix_no_gene_transposed['Diagnosis'].unique()

    for gene in tqdm.tqdm(gene_matrix_no_gene_transposed.columns[:-1]):  # Exclude 'Diagnosis' column
        gene_data = gene_matrix_no_gene_transposed[gene]

        for group in diagnosis_levels:
            # Group 1: current diagnosis group
            group_data = gene_data[gene_matrix_no_gene_transposed['Diagnosis'] == group]

            # Group 2: all other diagnosis groups combined
            rest_data = gene_data[gene_matrix_no_gene_transposed['Diagnosis'] != group]

            # Perform the Kolmogorov-Smirnov test
            ks_stat, ks_p_value = ks_2samp(group_data, rest_data)

            # Append the results
            results.append({
                'Gene': gene,
                'Comparison': f"{group} vs Rest",
                'KS_stat': ks_stat,
                'P_value': ks_p_value
            })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Apply Benjamini-Hochberg correction PER COMPARISON GROUP
    results_df['Adjusted_P_value'] = np.nan  # Placeholder

    for comparison in results_df['Comparison'].unique():
        mask = results_df['Comparison'] == comparison
        _, corrected_p_values, _, _ = multipletests(results_df.loc[mask, 'P_value'], method='fdr_bh')
        results_df.loc[mask, 'Adjusted_P_value'] = corrected_p_values

    # Save results to a CSV file
    results_df.to_csv('ks_all_vs_rest_results.csv', index=False)
    print("All K-S test results saved to 'ks_all_vs_rest_results.csv'.")

    return results_df

# Run the test
ks_results = ks_test_all_vs_rest(gene_matrix_no_gene_transposed)
