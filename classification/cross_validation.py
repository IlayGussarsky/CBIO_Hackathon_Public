import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import globals

from knn_classifier import KNNClassifier
from mean_ensemble_classifier import MeanClassifier
from random_forest import RandomForestModel


def evaluate_all_models(num_features_list):
    gene_order_methods = {
        "t-test": globals.gene_order_ttest,
        "ks test": globals.gene_order_kstest,
        "gmm test": globals.gene_order_gmm,
        "FNN": globals.gene_order_fnn
    }

    classifiers = {
        "KNN": KNNClassifier,
        "MeanEnsemble": MeanClassifier,
        "RandomForest": RandomForestModel
    }

    healthy, sick, feature_names = globals.healthy, globals.sick, globals.feature_names
    X = np.vstack((healthy, sick))
    y = np.hstack((np.zeros(len(healthy)), np.ones(len(sick))))

    results_combination = []
    results_selection = []

    for method_name, gene_order in tqdm(gene_order_methods.items(), desc="Feature Selection Methods"):
        for k in num_features_list:
            selected_genes = gene_order[:k]

            best_auc = -np.inf
            best_model = None
            best_method = None

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for clf_name, ClfClass in classifiers.items():
                auc_scores = []
                for train_index, test_index in skf.split(X, y):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    healthy_train = X_train[y_train == 0]
                    sick_train = X_train[y_train == 1]

                    model = ClfClass(genes=selected_genes)
                    model.fit(healthy_train, sick_train)

                    healthy_predictions = model.predict(X_test[y_test == 0])
                    sick_predictions = model.predict(X_test[y_test == 1])

                    y_true = np.hstack((np.zeros(len(healthy_predictions)), np.ones(len(sick_predictions))))
                    y_scores = np.hstack((healthy_predictions, sick_predictions))

                    auc = roc_auc_score(y_true, y_scores)
                    auc_scores.append(auc)

                avg_auc = np.mean(auc_scores)
                if avg_auc > best_auc:
                    best_auc = avg_auc
                    best_model = clf_name
                    best_method = method_name

                results_combination.append({
                    "Model": clf_name,
                    "Number of Features": k,
                    "Best AUC": avg_auc,
                    "Feature Selection Method": method_name
                })

            results_selection.append({
                "Feature Selection": best_method,
                "Number of Features": k,
                "Best AUC": best_auc,
                "Best Model": best_model
            })

    df_combination = pd.DataFrame(results_combination)
    df_selection = pd.DataFrame(results_selection)

    df_combination.to_csv("model_feature_auc_results.csv", index=False)
    df_selection.to_csv("feature_selection_auc_results.csv", index=False)


def plot_auc_heatmap_combination():
    df = pd.read_csv("cross validation results/model_feature_auc_results.csv")

    # Pivot table to get max AUC values
    pivot_auc = df.pivot_table(index='Model', columns='Number of Features', values='Best AUC', aggfunc='max')

    # Find the feature selection method that corresponds to the max AUC
    pivot_method = df.loc[df.groupby(['Model', 'Number of Features'])['Best AUC'].idxmax(),
                          ['Model', 'Number of Features', 'Feature Selection Method']].pivot(index='Model',
                          columns='Number of Features', values='Feature Selection Method')

    # Format AUC values to 4 decimal places and combine with feature selection method
    annot_data = pivot_auc.applymap(lambda x: f"{x:.4f}") + "\n" + pivot_method.astype(str)

    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(pivot_auc, annot=annot_data, cmap='viridis', fmt="", linewidths=0.5)
    plt.title("Heatmap of Best AUC by Classifier and Feature Count")
    plt.xlabel("Number of Features")
    plt.ylabel("Classifier")
    plt.show()


def plot_auc_heatmap_selection():
    df = pd.read_csv("cross validation results/feature_selection_auc_results.csv")

    # Pivot table to get max AUC values
    pivot_auc = df.pivot_table(index='Feature Selection', columns='Number of Features', values='Best AUC', aggfunc='max')

    # Find the best model corresponding to the max AUC
    pivot_model = df.loc[df.groupby(['Feature Selection', 'Number of Features'])['Best AUC'].idxmax(),
                         ['Feature Selection', 'Number of Features', 'Best Model']].pivot(index='Feature Selection',
                         columns='Number of Features', values='Best Model')

    # Format AUC values to 4 decimal places and combine with best model
    annot_data = pivot_auc.applymap(lambda x: f"{x:.4f}") + "\n"

    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(pivot_auc, annot=annot_data, cmap='viridis', fmt="", linewidths=0.5)
    plt.title("Heatmap of Best AUC by Feature Selection and Feature Count")
    plt.xlabel("Number of Features")
    plt.ylabel("Feature Selection Method")
    plt.show()


if __name__ == '__main__':
    num_features_list = [20, 30, 40, 50, 60, 70, 100, 200, 500]
    #evaluate_all_models(num_features_list)

    plot_auc_heatmap_combination()
    plot_auc_heatmap_selection()
