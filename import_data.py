import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def preprocess_data(file_path):
    """
    Preprocess the data from a CSV file for machine learning.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    # Load the dataset
    data = pd.read_csv(file_path)

    # Keep only the relevant columns
    columns_to_keep = ['price', 'distance', 'surge_multiplier', 'cab_type']
    data = data[columns_to_keep]

    # Drop rows where the target variable 'price' is missing
    data = data.dropna(subset=['price'])

    # Apply KNN Imputation for missing input features
    imputer = KNNImputer(n_neighbors=5)
    data[['distance', 'surge_multiplier']] = imputer.fit_transform(data[['distance', 'surge_multiplier']])

    # Handle missing 'cab_type' by filling with the most frequent value
    if data['cab_type'].isnull().any():
        most_frequent_cab_type = data['cab_type'].mode()[0]
        data['cab_type'] = data['cab_type'].fillna(most_frequent_cab_type)

    return data