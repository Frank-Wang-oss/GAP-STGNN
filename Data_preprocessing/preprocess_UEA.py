import os
import numpy as np
import torch
from scipy.io import arff
import pandas as pd


def _parse_relational_arff(data):
    """Parse relational ARFF data into numpy arrays"""
    X_data = np.asarray(data[0])
    n_samples = len(X_data)
    X, y = [], []

    if X_data[0][0].dtype.names is None:
        for i in range(n_samples):
            X_sample = np.asarray(
                [X_data[i][name] for name in X_data[i].dtype.names[:-1]]  # Exclude label
            )
            X.append(X_sample.T)
            y.append(X_data[i][-1])  # Last column is label
    else:
        for i in range(n_samples):
            X_sample = np.asarray(
                [X_data[i][0][name] for name in X_data[i][0].dtype.names]
            )
            X.append(X_sample.T)
            y.append(X_data[i][1])

    X = np.asarray(X).astype('float32')  # Use float32 for memory efficiency
    y = np.asarray(y)

    try:
        y = y.astype('float64').astype('int64')
    except ValueError:
        y = np.array([yi.decode('utf-8') if isinstance(yi, bytes) else str(yi) for yi in y])

    return X, y


def load_arff_data(dataset_path: str, split: str, dataset_name: str):
    """
    Load ARFF file and return properly formatted features and labels

    Args:
        dataset_path: Path to dataset directory
        split: Either "TRAIN" or "TEST"

    Returns:
        X: 3D numpy array of shape (n_samples, n_channels, n_timesteps)
        y: 1D numpy array of labels
        label_map: Dictionary mapping string labels to integers
    """
    # Construct file path
    arff_file = os.path.join(dataset_path, f"{dataset_name}_{split}.arff")

    # Load ARFF file
    data = arff.loadarff(arff_file)

    # Parse using the relational ARFF parser
    X, y = _parse_relational_arff(data)

    # Create label mapping (convert bytes to string if needed)
    unique_labels = np.unique(y)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_int = np.array([label_map[label] for label in y])

    return X, y_int, label_map


def save_as_torch(X: np.ndarray, y: np.ndarray, label_map: dict, save_path: str):
    """
    Save data as PyTorch tensor file

    Args:
        X: 3D feature array (already properly formatted)
        y: 1D label array
        label_map: Dictionary mapping string labels to integers
        save_path: Path to save .pt file
    """
    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()
    print(X_tensor)
    print(y_tensor)
    # Save with metadata
    torch.save({
        "samples": X_tensor,
        "labels": y_tensor,
        "label_mapping": label_map,
        "n_channels": X.shape[1],
        "n_timesteps": X.shape[2]
    }, save_path)


def process_dataset(dataset_path: str, output_dir: str, dataset_name: str):
    """
    Args:
        dataset_path: Path to dataset directory containing ARFF files
        output_dir: Directory to save processed files
    """
    os.makedirs(output_dir, exist_ok=True)

    # try:
        # Process training data
    print("Processing training data...")
    X_train, y_train, label_map = load_arff_data(dataset_path, "TRAIN", dataset_name)
    train_path = os.path.join(output_dir, "train.pt")
    save_as_torch(X_train, y_train, label_map, train_path)
    print(f"Training data shape: {X_train.shape}")

    # Process test data
    print("\nProcessing test data...")
    X_test, y_test, _ = load_arff_data(dataset_path, "TEST", dataset_name)
    test_path = os.path.join(output_dir, "test.pt")
    save_as_torch(X_test, y_test, label_map, test_path)
    print(f"Test data shape: {X_test.shape}")

    # Print dataset info
    print("\nDataset information:")
    print(f"Number of classes: {len(label_map)}")
    print(f"Class mapping: {label_map}")
    print(f"Saved processed files to: {output_dir}")


    # except Exception as e:
    #     print(f"\nError processing dataset: {str(e)}")
    #     return False


if __name__ == "__main__":
    # Example usage
    multivariate_datasets = [
        "CharacterTrajectories",
        "Cricket",
        "EigenWorms",
        "Epilepsy",
        "ERing",
        "FingerMovements",
        "SelfRegulationSCP1",
        "SpokenArabicDigits",
    ]

    for dataset in multivariate_datasets:
        dataset_name = dataset
        DATASET_PATH = f"raw_datasets/{dataset_name}"
        OUTPUT_DIR = f"datasets/{dataset_name}"

        process_dataset(DATASET_PATH, OUTPUT_DIR, dataset_name)
