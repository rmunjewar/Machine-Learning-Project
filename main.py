import os
from import_data import preprocess_data

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
    except Exception as e:
        print(f"An error occurred during data preprocessing: {e}")

if __name__ == "__main__":
    main()