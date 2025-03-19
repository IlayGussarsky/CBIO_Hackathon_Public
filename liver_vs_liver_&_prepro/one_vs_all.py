import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the results DataFrame
gene_matrix_no_gene_transposed = pd.read_csv('/Users/hilleldravish/Library/CloudStorage/GoogleDrive-darvish.hillel@mail.huji.ac.il/My Drive/CBIO-HACK/Data/CleanUp/cleaned_llb_matrix.csv')
results_df = pd.read_csv('ks_all_vs_rest_results-filtered.csv')

# Sort by the lowest p-values
sorted_results = results_df.sort_values('P_value').head(100)

# Prepare data for plotting
selected_genes = sorted_results['Gene'].unique()

# Create box plots and density plots for each gene
for gene in selected_genes:
    plt.figure(figsize=(14, 6))

    # Box Plot
    plt.subplot(1, 2, 1)
    sns.boxplot(
        x='Diagnosis', y=gene, data=gene_matrix_no_gene_transposed,
        palette='Set2'
    )
    plt.title(f'Box Plot for {gene} (One vs. All)', fontsize=14)
    plt.xlabel('Diagnosis', fontsize=12)
    plt.ylabel('Expression Level', fontsize=12)

    # Density Plot
    plt.subplot(1, 2, 2)
    for diagnosis in gene_matrix_no_gene_transposed['Diagnosis'].unique():
        subset = gene_matrix_no_gene_transposed[gene_matrix_no_gene_transposed['Diagnosis'] == diagnosis]
        sns.kdeplot(subset[gene], label=f'{diagnosis}', fill=True, alpha=0.3)
    plt.title(f'Density Plot for {gene} (One vs. All)', fontsize=14)
    plt.xlabel('Expression Level', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(title='Diagnosis')

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(f'plots2/{gene}_box_density_plot.png')
    plt.show()
