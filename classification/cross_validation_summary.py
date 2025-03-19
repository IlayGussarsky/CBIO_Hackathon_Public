import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import globals
from knn_classifier import KNNClassifier
from mean_ensemble_classifier import MeanClassifier
from random_forest import RandomForestModel
from weighted_ll import MultiGeneClassifier, GeneClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

            best_f1 = -np.inf
            best_auc = -np.inf
            best_accuracy = -np.inf
            best_tpr = -np.inf  # Initialize best TPR
            best_model_f1 = None
            best_model_auc = None
            best_model_accuracy = None
            best_model_tpr = None  # Initialize best TPR model
            best_method = None

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for clf_name, ClfClass in classifiers.items():
                f1_scores = []
                auc_scores = []
                accuracy_scores = []
                tpr_scores = []  # List to store TPR scores

                for train_index, test_index in skf.split(X, y):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    healthy_train = X_train[y_train == 0]
                    sick_train = X_train[y_train == 1]

                    model = ClfClass(genes=selected_genes)
                    model.fit(healthy_train, sick_train)

                    predictions = model.predict(X_test)
                    # if model == MultiGeneClassifier:
                    #     probas = model.predict_proba(X_test)[:, 1]  # For AUC calculation
                    # else:
                    #     probas = predictions

                    # Compute F1-score, AUC, Accuracy, and TPR
                    f1 = f1_score(y_test, predictions.round())  # Ensure binary classification
                    auc = roc_auc_score(y_test, predictions)
                    accuracy = np.mean(predictions.round() == y_test)

                    # Calculate confusion matrix and TPR
                    tn, fp, fn, tp = confusion_matrix(y_test, predictions.round()).ravel()
                    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0  # Avoid division by zero

                    f1_scores.append(f1)
                    auc_scores.append(auc)
                    accuracy_scores.append(accuracy)
                    tpr_scores.append(tpr)  # Add TPR score

                avg_f1 = np.mean(f1_scores)
                avg_auc = np.mean(auc_scores)
                avg_accuracy = np.mean(accuracy_scores)
                avg_tpr = np.mean(tpr_scores)  # Average TPR score

                # Check for the best F1, AUC, accuracy, and TPR
                if avg_f1 > best_f1:
                    best_f1 = avg_f1
                    best_model_f1 = clf_name

                if avg_auc > best_auc:
                    best_auc = avg_auc
                    best_model_auc = clf_name

                if avg_accuracy > best_accuracy:
                    best_accuracy = avg_accuracy
                    best_model_accuracy = clf_name

                if avg_tpr > best_tpr:
                    best_tpr = avg_tpr  # Update best TPR
                    best_model_tpr = clf_name  # Update best TPR model

                results_combination.append({
                    "Model": clf_name,
                    "Number of Features": k,
                    "Best F1": avg_f1,
                    "Best AUC": avg_auc,
                    "Best Accuracy": avg_accuracy,
                    "Best TPR": avg_tpr,  # Include TPR in results
                    "Feature Selection Method": method_name
                })

            results_selection.append({
                "Feature Selection": best_method,
                "Number of Features": k,
                "Best F1": best_f1,
                "Best AUC": best_auc,
                "Best Accuracy": best_accuracy,
                "Best TPR": best_tpr,  # Include best TPR
                "Best Model F1": best_model_f1,
                "Best Model AUC": best_model_auc,
                "Best Model Accuracy": best_model_accuracy,
                "Best Model TPR": best_model_tpr  # Include best TPR model
            })

    df_combination = pd.DataFrame(results_combination)
    df_selection = pd.DataFrame(results_selection)

    # Save the results to CSV with new names
    df_combination.to_csv("model_feature_combination_results_v2.csv", index=False)
    df_selection.to_csv("feature_selection_summary_results_v2.csv", index=False)


def plot_maximum_metrics(csv_file="model_feature_combination_results_v2.csv"):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Extract maximum values for each classifier and evaluation method
    best_f1 = df.groupby("Model")["Best F1"].max()
    best_auc = df.groupby("Model")["Best AUC"].max()
    best_accuracy = df.groupby("Model")["Best Accuracy"].max()
    best_tpr = df.groupby("Model")["Best TPR"].max()

    # Create a new DataFrame for the metrics
    metrics = pd.DataFrame({
        'Best F1': best_f1,
        'Best AUC': best_auc,
        'Best Accuracy': best_accuracy,
        'Best TPR': best_tpr
    })

    # Create annotations (Feature Selection Method and Number of Features)
    annotations = pd.DataFrame(index=metrics.index, columns=metrics.columns)
    for metric in metrics.columns:
        for i, model in enumerate(metrics.index):
            best_row = df[(df['Model'] == model) & (df[metric] == metrics.loc[model, metric])]
            feature_selection_method = best_row["Feature Selection Method"].values[0]
            num_features = best_row["Number of Features"].values[0]
            score = metrics.loc[model, metric]
            annotations.loc[model, metric] = f"{score:.3f}\n{feature_selection_method} ({num_features} features)"

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(metrics, annot=annotations, fmt="", cmap="coolwarm", linewidths=0.5, cbar_kws={'label': 'Metric Value'})

    plt.title("Maximum Metric Values by Classifier")
    plt.xlabel("Metric")
    plt.ylabel("Classifier")

    plt.show()


def summarize_and_plot_best_metrics(csv_file="model_feature_combination_results_v2.csv"):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Define evaluation metrics
    metrics = ["Best F1", "Best AUC", "Best Accuracy", "Best TPR"]

    # Create a summary table
    summary = []
    for model in df["Model"].unique():
        row = {"Model": model}
        model_df = df[df["Model"] == model]  # Filter by model first
        for metric in metrics:
            if model_df[metric].isnull().all():
                continue  # Skip if all values are NaN
            best_idx = model_df[metric].idxmax()  # Find best score index in filtered dataframe
            best_row = model_df.loc[best_idx]  # Select row
            row[metric] = best_row[metric]
            row[f"Feature Selection ({metric})"] = best_row["Feature Selection Method"]
            row[f"Number of Features ({metric})"] = best_row["Number of Features"]
        summary.append(row)

    summary_df = pd.DataFrame(summary)

    # Plot the summary table
    fig, ax = plt.subplots(figsize=(14, len(summary_df) * 0.5))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(cellText=summary_df.values,
                     colLabels=summary_df.columns,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([i for i in range(len(summary_df.columns))])

    plt.show()

    return summary_df


if __name__ == '__main__':
    num_features_list = [20, 30, 40, 50, 60, 70, 100, 200, 500]
    #evaluate_all_models(num_features_list)
    plot_maximum_metrics()
    #summarize_and_plot_best_metrics()
