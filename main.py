import os
from import_data import preprocess_data
from knn_regression import train_knn_regression, validate_knn_model, test_knn_model

def main():
    # Define the path to the dataset relative to the project directory
    project_root = os.path.dirname(__file__)
    dataset_path = os.path.join(project_root, 'dataset', 'cab_rides.csv')

    # Check if the file exists
    if not os.path.exists(dataset_path):
        print(f"Error: The dataset file '{dataset_path}' does not exist.")
        print("Please download and add the file to the specified directory before continuing.")
        return

    # Preprocess the data
    try:
        preprocessed_data = preprocess_data(dataset_path)
        print("Data preprocessing completed successfully.")
        model = train_knn_regression(preprocessed_data)
        print("KNN Regression model training completed successfully.")
        tuned_model = validate_knn_model(model, preprocessed_data)
        print("KNN model validation and hyperparameter tuning completed successfully.")
        test_knn_model(tuned_model, preprocessed_data)
        print("Model evaluation completed successfully.")
    except Exception as e:
        print(f"An error occurred during data preprocessing: {e}")

if __name__ == "__main__":
    main()