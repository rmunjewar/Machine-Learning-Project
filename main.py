import os
from sklearn.model_selection import train_test_split
from gradient_boosting import train_gradient_boosting, tune_gradient_boosting
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
        # Overall
        preprocessed_data = preprocess_data(dataset_path)
        print("-------------------------------------------------------\n")
        print("NOTE: Data preprocessing completed successfully.\n")
        print("-------------------------------------------------------\n")

        # Split the data into training, validation, and testing sets
        train_data, temp_data = train_test_split(preprocessed_data, test_size=0.3, random_state=42)
        validation_data, test_data = train_test_split(temp_data, test_size=0.3, random_state=42)

        print(f"Training set size: {len(train_data)}")
        print(f"Validation set size: {len(validation_data)}")
        print(f"Testing set size: {len(test_data)}")
        print("-------------------------------------------------------\n")

        # KNN Model
        # print("KNN MODEL:\n")
        # model = train_knn_regression(train_data)
        # print("NOTE: KNN Regression model training completed successfully.\n")
        # print("-------------------------------------------------------\n")
        # tuned_model = validate_knn_model(model, validation_data)
        # print("KNN model validation and hyperparameter tuning completed successfully.")
        # test_knn_model(tuned_model, test_data)
        # print("Model evaluation completed successfully.")
        # print("-------------------------------------------------------\n")

        # Gradient Boosting
        print("GRADIENT BOOSTING:\n")
        model = train_gradient_boosting(train_data)
        print("NOTE: Gradient Boosting model training completed successfully.\n")
        best_model = tune_gradient_boosting(model, validation_data)
        print("-------------------------------------------------------\n")

        # Random Forest Model
        # print("RANDOM FOREST MODEL:\n")
        # model = train_random_forest(train_data)
        # print("NOTE: Random Forest Model training completed successfully.\n")
        # print("-------------------------------------------------------\n")
        # tuned_model = validate_random_forest(model, validation_data)
        # print("Random Forest Model validation and hyperparameter tuning completed successfully.")
    

        # Linear Regression Model
        # print("LINEAR REGRESSION MODEL:\n")
        # model = train_linear_regression(train_data)
        # print("NOTE: Linear Regression model training completed successfully.\n")
        # print("-------------------------------------------------------\n")
        # test_linear_regression(tuned_model, test_data)
        # print("Linear Regression model evaluation completed successfully.")


    except Exception as e:
        print(f"An error occurred during data preprocessing: {e}")

if __name__ == "__main__":
    main()