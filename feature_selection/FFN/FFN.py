import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import optuna
import argparse

def load_and_preprocess_data(healthy_path, sick_path, additional_healthy_path):
    # Load the CSV files
    healthy_df = pd.read_csv(healthy_path, index_col=0)
    sick_df = pd.read_csv(sick_path, index_col=0)
    additional_healthy_df = pd.read_csv(additional_healthy_path, index_col=0)

    # Transpose the data
    healthy_df = healthy_df.T
    sick_df = sick_df.T
    additional_healthy_df = additional_healthy_df.T

    # Add labels
    healthy_df['label'] = 0
    sick_df['label'] = 1
    additional_healthy_df['label'] = 0

    # Combine the datasets
    combined_df = pd.concat([healthy_df, sick_df])

    # Remove rows with NaN values
    combined_df = combined_df.dropna()
    additional_healthy_df = additional_healthy_df.dropna()

    # Shuffle the dataset
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Separate features and labels
    X = combined_df.drop(columns=['label']).values
    y = combined_df['label'].values

    # Split the data into train+val (80%) and test (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Split the train+val set into train (60%) and validation (20%)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Prepare additional healthy data for testing
    X_additional_healthy = additional_healthy_df.drop(columns=['label']).values
    X_additional_healthy = scaler.transform(X_additional_healthy)
    y_additional_healthy = additional_healthy_df['label'].values

    return X_train, X_val, X_test, y_train, y_val, y_test, X_additional_healthy, y_additional_healthy

def objective(trial, X_train, y_train, X_val, y_val):
    # Define hyperparameters to optimize
    num_layers = trial.suggest_int('num_layers', 1, 5)
    layer_width = trial.suggest_int('layer_width', 16, 256)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    # Build the model
    model = Sequential()
    model.add(Dense(layer_width, activation='relu', input_shape=(X_train.shape[1],)))

    # Add hidden layers
    for _ in range(num_layers - 1):
        model.add(Dense(layer_width, activation='relu'))

    # Add output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model with validation data
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), verbose=0)

    # Evaluate the model on the validation set
    _, accuracy = model.evaluate(X_val, y_val, verbose=0)

    return accuracy

from sklearn.metrics import roc_curve, auc

def evaluate_and_plot(model, X, y, title, filename, new_dir, file=None):
    # Predict on the data
    y_pred = model.predict(X)
    y_pred_classes = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

    # Confusion Matrix
    conf_matrix = confusion_matrix(y, y_pred_classes)
    print(f"Confusion Matrix ({title}):", file=file)
    print(conf_matrix, file=file)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Healthy', 'Sick'], yticklabels=['Healthy', 'Sick'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix ({title})')
    plt.savefig(f"{new_dir}/confusion_matrix_{filename}.png")  # Save confusion matrix plot
    plt.close()

    # Classification Report
    print(f"Classification Report ({title}):", file=file)
    print(classification_report(y, y_pred_classes, target_names=['Healthy', 'Sick']), file=file)

    # Additional Metrics
    accuracy = accuracy_score(y, y_pred_classes)
    precision = precision_score(y, y_pred_classes)
    recall = recall_score(y, y_pred_classes)
    f1 = f1_score(y, y_pred_classes)

    print(f"Accuracy: {accuracy:.4f}", file=file)
    print(f"Precision: {precision:.4f}", file=file)
    print(f"Recall: {recall:.4f}", file=file)
    print(f"F1-Score: {f1:.4f}", file=file)
    print("\n", file=file)

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y, y_pred)  # Calculate FPR, TPR, and thresholds
    roc_auc = auc(fpr, tpr)  # Calculate the area under the ROC curve


    print(f"ROC AUC Score ({title}): {roc_auc:.4f}\n", file=file)



def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a neural network model for liver cancer classification.")
    parser.add_argument('--healthy', type=str, required=True, help="Path to the healthy samples CSV file.")
    parser.add_argument('--sick', type=str, required=True, help="Path to the sick samples CSV file.")
    parser.add_argument('--additional_healthy', type=str, required=True,
                        help="Path to the additional healthy samples CSV file.")
    parser.add_argument('--n_trials', type=int, default=5,
                        help="Number of Optuna trials for hyperparameter optimization.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train the final model.")
    parser.add_argument('--min_layer', type=int, default=1, help="Min layer amount")
    parser.add_argument('--max_layer', type=int, default=1, help="Max layer amount")
    parser.add_argument('--try_int', type=int, default=1, help="Int of the try number of this learning and testing")
    args = parser.parse_args()

    new_dir = f'try_{args.try_int}_trials_{args.n_trials}_epochs_{args.epochs}'
    os.mkdir(new_dir)

    # Open a text file to write all outputs
    with open(f'{new_dir}/output_log_{args.try_int}.txt', 'w') as file:
        # Load and preprocess data
        X_train, X_val, X_test, y_train, y_val, y_test, X_additional_healthy, y_additional_healthy = load_and_preprocess_data(
            args.healthy, args.sick, args.additional_healthy
        )

        # Create an Optuna study and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=args.n_trials)

        # Print and write the best hyperparameters
        print("Best Hyperparameters:", study.best_params, file=file)
        print("Best Hyperparameters:", study.best_params)

        # Build the final model with the best hyperparameters
        best_params = study.best_params
        final_model = Sequential()
        final_model.add(Dense(best_params['layer_width'], activation='relu', input_shape=(X_train.shape[1],)))

        # Add hidden layers
        for _ in range(best_params['num_layers'] - 1):
            final_model.add(Dense(best_params['layer_width'], activation='relu'))

        # Add output layer
        final_model.add(Dense(1, activation='sigmoid'))

        # Compile the final model
        final_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']),
                            loss='binary_crossentropy',
                            metrics=['accuracy'])

        # Train the final model with validation data
        final_model.fit(X_train, y_train, epochs=args.epochs, batch_size=32, validation_data=(X_val, y_val))

        # Evaluate on the training set
        evaluate_and_plot(final_model, X_train, y_train, "Training Set", "train_set", new_dir, file=file)

        # Evaluate on the validation set
        evaluate_and_plot(final_model, X_val, y_val, "Validation Set", "val_set", new_dir, file=file)

        # Evaluate on the test set
        evaluate_and_plot(final_model, X_test, y_test, "Test Set", "test_set", new_dir, file=file)

        # Evaluate on the additional healthy data alone
        evaluate_and_plot(final_model, X_additional_healthy, y_additional_healthy, "Additional Healthy Set", "additional_healthy_set", new_dir, file=file)

        # Save the final model
        final_model.save(f'{new_dir}/final_nn_model.h5')


if __name__ == "__main__":
    main()