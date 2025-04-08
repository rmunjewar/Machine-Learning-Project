import os
from import_data import preprocess_data
from knn_regression import train_knn_regression, validate_knn_model, test_knn_model
from random_forest import train_random_forest, validate_random_forest
from linear_regression import train_linear_regression, test_linear_regression


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

         # Train and evaluate KNN
        print("KNN Model")
        model_knn = train_knn_regression(preprocessed_data)
        tuned_knn = validate_knn_model(model_knn, preprocessed_data)
        test_knn_model(tuned_knn, preprocessed_data)

        # Train and evaluate Random Forest
        print("Random Forest Model")
        model_rf = train_random_forest(preprocessed_data)
        tuned_rf = validate_random_forest(model_rf, preprocessed_data)

        # Linear REgression
        print("Linear Regression Model")
        model_lr = train_linear_regression(preprocessed_data)
        test_linear_regression(model_lr, preprocessed_data)


    except Exception as e:
        print(f"An error occurred during data preprocessing: {e}")

if __name__ == "__main__":
    main()