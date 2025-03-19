import re
import os

def parse_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Initialize the dictionary
    data = {}

    # Extract hyperparameters
    hyperparams_match = re.search(r"Best Hyperparameters: ({.*})", content)
    if hyperparams_match:
        hyperparams = eval(hyperparams_match.group(1))
        data.update(hyperparams)

    # Extract metrics using regex
    metrics = {
        "train_accuracy": r"Accuracy: ([\d.]+)",
        "train_precision": r"Precision: ([\d.]+)",
        "train_recall": r"Recall: ([\d.]+)",
        "train_f1-Score": r"F1-Score: ([\d.]+)",
        "train_roc_auc": r"ROC AUC Score \(Training Set\): ([\d.]+)",
        "validation_accuracy": r"Confusion Matrix \(Validation Set\):.*?Accuracy: ([\d.]+)",
        "validation_precision": r"Confusion Matrix \(Validation Set\):.*?Precision: ([\d.]+)",
        "validation_recall": r"Confusion Matrix \(Validation Set\):.*?Recall: ([\d.]+)",
        "validation_f1-Score": r"Confusion Matrix \(Validation Set\):.*?F1-Score: ([\d.]+)",
        "validation_roc_auc": r"ROC AUC Score \(Validation Set\): ([\d.]+)",
        "test_accuracy": r"Confusion Matrix \(Test Set\):.*?Accuracy: ([\d.]+)",
        "test_precision": r"Confusion Matrix \(Test Set\):.*?Precision: ([\d.]+)",
        "test_recall": r"Confusion Matrix \(Test Set\):.*?Recall: ([\d.]+)",
        "test_f1-Score": r"Confusion Matrix \(Test Set\):.*?F1-Score: ([\d.]+)",
        "test_roc_auc": r"ROC AUC Score \(Test Set\): ([\d.]+)",
        "additional_accuracy": r"Confusion Matrix \(Additional Healthy Set\):.*?Accuracy: ([\d.]+)",
    }

    for key, pattern in metrics.items():
        match = re.search(pattern, content, re.DOTALL)  # Use re.DOTALL to match across lines
        if match:
            data[key] = float(match.group(1))

    return data

def main(paths_file):
    # Read the file containing absolute paths
    with open(paths_file, 'r') as file:
        absolute_paths = file.read().splitlines()

    # Initialize the main dictionary to store all results
    results = {}

    # Process each file
    for absolute_path in absolute_paths:
        # Get the file name (without the absolute path)
        file_name = os.path.basename(absolute_path)
        # Parse the file and get the data
        file_data = parse_file(absolute_path)
        # Add the data to the results dictionary with the file name as the key
        results[file_name] = file_data

    return results

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_boxplots(results):
    # Initialize lists to store data for the DataFrame
    data = []

    # Iterate over each file's results
    for file_name, metrics in results.items():
        for metric, value in metrics.items():
            # Skip hyperparameters
            if metric in ["num_layers", "layer_width", "learning_rate"]:
                continue

            # Determine the category (train, validation, test, additional)
            if metric.startswith("train"):
                category = "train"
            elif metric.startswith("validation"):
                category = "validation"
            elif metric.startswith("test"):
                category = "test"
            elif metric.startswith("additional"):
                category = "additional"
            else:
                continue

            # Append the data to the list
            data.append({
                "Category": category,
                "Metric": metric,
                "Value": value
            })

    # Convert the data to a DataFrame
    df = pd.DataFrame(data)

    # Define a color palette for the categories
    palette = {
        "train": sns.color_palette("Blues", n_colors=4)[-1],  # Dark blue
        "validation": sns.color_palette("Greens", n_colors=4)[-1],  # Dark green
        "test": sns.color_palette("Reds", n_colors=4)[-1],  # Dark red
        "additional": sns.color_palette("YlOrBr", n_colors=4)[-1],  # Dark yellow
    }

    # Create the box plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Metric", y="Value", hue="Category", data=df, palette=palette)

    # Customize the plot
    plt.title("Box Plot of Metrics by Category", fontsize=16)
    plt.xlabel("Metric", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
    plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")  # Move legend outside
    plt.tight_layout()  # Adjust layout to prevent overlap
    fnn_results_box_plot = "fnn_results_box_plot.png"
    plt.savefig(fnn_results_box_plot)
    plt.close()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_scatter(results, output_path="scatter_plot.png"):
    # Initialize lists to store data for the DataFrame
    data = []

    # Iterate over each file's results
    for file_name, metrics in results.items():
        # Extract the required metrics
        num_layers = metrics.get("num_layers")
        layer_width = metrics.get("layer_width")
        learning_rate = metrics.get("learning_rate")

        # Append the data to the list
        if num_layers is not None and layer_width is not None and learning_rate is not None:
            data.append({
                "num_layers": num_layers,
                "layer_width": layer_width,
                "learning_rate": learning_rate
            })

    # Convert the data to a DataFrame
    df = pd.DataFrame(data)

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(
        x="num_layers",
        y="layer_width",
        hue="learning_rate",  # Color by learning_rate
        data=df,
        palette="viridis",  # Use the viridis color palette
        size=None,  # Ensure all dots have the same size
        legend=False  # Remove the legend for learning_rate
    )

    # Add a color bar using ScalarMappable
    norm = plt.Normalize(df["learning_rate"].min(), df["learning_rate"].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])  # Set an empty array for ScalarMappable
    cbar = plt.colorbar(sm, ax=plt.gca())  # Attach the color bar to the current Axes
    cbar.set_label("Learning Rate", fontsize=12)

    # Customize the plot
    plt.title("Scatter Plot of num_layers vs layer_width (Colored by learning_rate)", fontsize=16)
    plt.xlabel("Number of Layers (num_layers)", fontsize=14)
    plt.ylabel("Layer Width (layer_width)", fontsize=14)
    plt.xticks(ticks=range(int(df["num_layers"].min()), int(df["num_layers"].max()) + 1))  # Ensure x-axis values are integers
    plt.grid(True, linestyle="--", alpha=0.7)  # Add a grid for better readability
    plt.tight_layout()  # Adjust layout to prevent overlap

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()







# Example usage
if __name__ == "__main__":
    # Path to the file containing absolute paths
    paths_file = "/cs/zbio/tommy/CBIO2025/liver_cancer_group/code/logs_paths.txt"  # Replace with the actual path to your file
    # Get the results
    results = main(paths_file)
    # Print the results
    print(results)
    plot_scatter(results)