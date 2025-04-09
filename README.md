# Machine-Learning-Project #

## Setting up the project ##

1. **Download the dataset**  
    Download the data from [KaggleHub](https://www.kaggle.com/datasets/ravi72munde/uber-lyft-cab-prices?resource=download) and save the dataset  into a folder called `dataset` in the root directory of this project. The dataset file should be named `cab_rides.csv`.

2. **Install dependencies**  
    Run the following command to install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the project**
    Execute the `main.py` file to pre-process the data (have not started on models yet):

4. **Error Handling**
    If the dataset file is missing `dataset/cab_rides.csv`, the program will notify you to download the file manually. This is done because the files are too large for GitHub to upload correctly.


## Project Overview ##

    This project aims to predict the price of a cab ride using a machine learning pipeline trained on ride data from Lyft and Uber. This task falls under supervised regression, where the model learns patterns between input features (like distance, cab type, and surge multiplier) and the continuous target variable (price).

    By implementing, training, tuning, and evaluating multiple models, we aim to identify the most accurate and generalizable approach for price prediction.

1. **Preprocessing**
    * Removes all rows with missing output feature *price*
    * scales the numerical fields *distance* and surge_multiplier*
    * Uses KNN Imputation to handle missing input input features *distance*, and *surge_multiplier*
    * Filles in missing data for the *cab_type* data with the most frequent value.

    KNN Imputation is used on the more important fields for determining the price of the cab trip, while a more simple method of selecting the mode works for the most frequent value to prevent the program from becoming cumbersome.

2. **Models Implemented**
    Each model is trained on the processed dataset and validated using GridSearchCV where applicable.

    **Linear Regression** 
       - Implements a simple baseline model.
       - Outputs Mean Squared Error (MSE) and R² score for training and test data.

    **K-Nearest Neighbors (KNN)**
       - Predicts using feature similarity.
       - Hyperparameters tuned:
            n_neighbors (5–20)
            weights (uniform, distance)
       - GridSearchCV used for optimal configuration.
       - Outputs training MSE, R², and best parameters

    **Random Forest Regressor**
       - An ensemble method using decision trees.
       - Hyperparameters tuned:
            n_estimators (50–200)
            max_depth (None, 10, 20)
            min_samples_split (2, 5)

    **Gradient Boosting Regressor**
       - Boosted ensemble of shallow trees.
       - Hyperparameters tuned:
            n_estimators (50–250)
            learning_rate (0.01–0.3)
       - Outputs negative MSE and best parameters 


3. **Performance Evaluation**
   - All models are evaluated using:
        Mean Squared Error (MSE)
        R² Score
        Cross-validation on validation set
   - Evaluation occurs after each model is trained and hyperparameters are tuned. 