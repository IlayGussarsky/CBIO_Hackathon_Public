import os
from tensorflow.keras.models import load_model
import numpy as np
import re

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

def calculate_weighted_average(weights_first_layer, weights_second_layer):
    """
    Calculates the weighted average of the first layer's weights using the second layer's weights.

    Args:
        weights_first_layer (numpy.ndarray): Weights of the first layer (shape: [input_dim, neurons_in_first_layer]).
        weights_second_layer (numpy.ndarray): Weights of the second layer (shape: [neurons_in_first_layer, 1]).

    Returns:
        numpy.ndarray: Weighted average vector of the first layer's weights.
    """
    # Ensure shapes are compatible
    if weights_first_layer.shape[1] != weights_second_layer.shape[0]:
        raise ValueError("Mismatch between first and second layer dimensions.")

    # Reshape second layer weights to 1D array
    weights_second_layer = weights_second_layer.flatten()

    # Calculate the weighted average (sum of 112 vectors, each scaled by the second layer's weight)
    weighted_avg = np.dot(weights_first_layer, weights_second_layer)

    return weighted_avg

def calculate_weighted_average_from_dict(weighted_avg_vector_and_test_roc_auc_dict):
    """
    Calculates a weighted average of all weighted_avg_vector values in the dictionary,
    using their corresponding test_roc_auc values as weights.

    Args:
        weighted_avg_vector_and_test_roc_auc_dict (dict):
            Dictionary where keys are paths, and values are [test_roc_auc, weighted_avg_vector].

    Returns:
        numpy.ndarray: Weighted average vector.
    """
    # Initialize numerator and denominator for weighted average calculation
    numerator = 0
    denominator = 0

    for path, (test_roc_auc, weighted_avg_vector) in weighted_avg_vector_and_test_roc_auc_dict.items():
        numerator += test_roc_auc * np.array(weighted_avg_vector)  # Add weighted vector
        denominator += test_roc_auc  # Add weight

    # Avoid division by zero
    if denominator == 0:
        raise ValueError("Sum of test_roc_auc values is zero. Cannot calculate weighted average.")

    # Calculate the weighted average vector
    weighted_avg = numerator / denominator
    return weighted_avg


def analyze_model(path):
    """
    Analyzes a single model to determine its layer count and calculate the weighted average if it has 2 layers.

    Args:
        path (str): Path to the .h5 model file.
    """
    try:
        # Load the model
        dirname = os.path.dirname(path)
        print(dirname.split("_"))
        try_int = int(dirname.split("_")[-5])
        print("try_int", try_int)
        path_for_log = dirname+f"/output_log_{try_int}.txt"
        model_data = parse_file(path_for_log)
        layer_count = model_data['num_layers']
        test_roc_auc = model_data['test_roc_auc']
        print("test_roc_auc", test_roc_auc)
        print(f"Model at {path} has {layer_count} layers.")

        if layer_count == 1:
            print("layer count 2")
            model = load_model(path)
            # Extract weights from the first and second layers
            first_layer_weights, _ = model.layers[0].get_weights()
            second_layer_weights, _ = model.layers[1].get_weights()

            print("First layer weights shape:", first_layer_weights.shape)
            print("Second layer weights shape:", second_layer_weights.shape)

            # Calculate weighted average
            weighted_avg_vector = calculate_weighted_average(first_layer_weights, second_layer_weights)

            return (test_roc_auc, weighted_avg_vector)
        else:
            print(f"Skipping calculation for model at {path} as it does not have exactly 2 layers.")
    except Exception as e:
        print(f"Error processing model at {path}: {e}")
    return (None, None)




def make_averaged_averaged_weights(file_path):
    """
    Reads paths to .h5 files from a file and analyzes each model.

    Args:
        file_path (str): Path to the file containing .h5 model paths, one per line.
    """
    weighted_avg_vector_and_test_roc_auc_dict = dict()
    with open(file_path, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]  # Read and clean paths

    for path in paths:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        print("some path", path)
        test_roc_auc, weighted_avg_vector = analyze_model(path)
        print("test_roc_auc", test_roc_auc)
        print("weighted_avg_vector", weighted_avg_vector)
        if test_roc_auc==None:
            continue
        weighted_avg_vector_and_test_roc_auc_dict[path] = [test_roc_auc, weighted_avg_vector]
    print(weighted_avg_vector_and_test_roc_auc_dict)
    weighted_avg = calculate_weighted_average_from_dict(weighted_avg_vector_and_test_roc_auc_dict)
    return weighted_avg

import pandas as pd
import numpy as np

import pandas as pd

def save_sorted_features(weighted_avg_vector, csv_path):
    """
    Combines a weighted average vector with gene names from a CSV file,
    sorts them by the absolute value of the weights, and saves to a new CSV.

    Args:
        weighted_avg_vector (numpy.ndarray): The weighted average vector (values correspond to gene names).
        csv_path (str): Path to the input CSV file (must have a column named 'GeneName').

    Raises:
        ValueError: If the length of the weighted_avg_vector does not match the number of rows in the CSV.
    """
    # Read the input CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file at {csv_path}: {e}")

    # Check if 'GeneName' column exists
    if 'GeneName' not in df.columns:
        raise ValueError("The CSV file must contain a column named 'GeneName'.")

    # Validate the length of the vector
    if len(weighted_avg_vector) != len(df):
        raise ValueError(
            "The length of the weighted_avg_vector does not match the number of rows in the CSV file."
        )

    # Create a new DataFrame with only the 'GeneName' and 'Weight' columns
    df_result = df[['GeneName']].copy()
    df_result['Weight'] = weighted_avg_vector

    # Save the unsorted version
    unsorted_output_path = "FNN_features_unsorted.csv"
    df_result.to_csv(unsorted_output_path, index=False)
    print(f"Unsorted features saved to {unsorted_output_path}")

    # Sort by the absolute value of the weights
    df_sorted = df_result.reindex(df_result['Weight'].abs().sort_values(ascending=False).index)

    # Save the sorted version
    sorted_output_path = "FNN_features_sorted.csv"
    df_sorted.to_csv(sorted_output_path, index=False)
    print(f"Sorted features saved to {sorted_output_path}")





def main():
    model_paths_txt = "/cs/zbio/tommy/CBIO2025/liver_cancer_group/code/model_paths.txt"
    data_csv = "/cs/zbio/tommy/CBIO2025/liver_cancer_group/Data/GeneMatrix/GeneMatrix-LLB.csv"

    weighted_avg = make_averaged_averaged_weights(model_paths_txt)
    save_sorted_features(weighted_avg, data_csv)


if __name__ == "__main__":
    main()
