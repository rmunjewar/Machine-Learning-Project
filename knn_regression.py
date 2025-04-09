import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_knn_regression(data: pd.DataFrame) -> KNeighborsRegressor:
    """
    Train a KNN Regression model to predict price based on distance, surge_multiplier, and cab_type.

    Args:
        data (pd.DataFrame): Preprocessed and scaled dataframe.

    Returns:
        KNeighborsRegressor: Trained KNN model.
    """
    data = pd.get_dummies(data, columns=['cab_type'], drop_first=True)

    X = data.drop(columns=['price'])
    y = data['price']
    
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X, y)

    return knn

def validate_knn_model(model: KNeighborsRegressor, data: pd.DataFrame) -> KNeighborsRegressor:
    """
    Validate the KNN model on a new dataset and tune hyperparameters.

    Args:
        model (KNeighborsRegressor): Trained KNN model.
        data (pd.DataFrame): Preprocessed and scaled dataframe for validation.

    Returns:
        KNeighborsRegressor: Best KNN model after hyperparameter tuning.
    """

    # Seperate features and target
    X = data.drop(columns=['price'])
    y = data['price']

    param_grid = {'n_neighbors': range(5, 21), 'weights': ['uniform', 'distance']}
    grid_search = GridSearchCV(
        estimator = model, 
        param_grid = param_grid,
        cv=5,
        scoring='neg_mean_squared_error'
    )
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Print the best parameters and performance
    print(f"Best Hyperparameters: {best_params}")
    print(f"Best Negative Mean Squared Error: {grid_search.best_score_:.2f}")

    return best_model
