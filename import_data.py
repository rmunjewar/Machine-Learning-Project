import pandas as pd
import kagglehub
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(file_path):
    """
    Preprocess the data from a CSV file for machine learning.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    # Load the dataset
    path = kagglehub.dataset_download("ravi72munde/uber-lyft-cab-prices")
    data = pd.read_csv(path)

    # Keep only the relevant columns
    columns_to_keep = ['price', 'distance', 'surge_multiplier', 'cab_type', 'product_id']
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

    # scaler - scale numerical features (distance, surge_multiplier)
    scaler = StandardScaler()
    data[['distance', 'surge_multiplier']] = scaler.fit_transform(data[['distance', 'surge_multiplier']])

    # encode the categorical 'cab_type' feature
    le = LabelEncoder()
    data['cab_type'] = le.fit_transform(data['cab_type'])

    if data['product_id'].dtype == 'object':  # for categorical
        data['product_id'] = le.fit_transform(data['product_id'])


    return data