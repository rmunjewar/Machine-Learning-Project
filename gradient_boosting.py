import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

def train_gradient_boosting(data: pd.DataFrame) -> GradientBoostingRegressor:
    """
    Trains a GradientBoostingRegressor to predict ride price based on input features.

    Parameters:
    - data (pd.DataFrame): A DataFrame that includes pre-encoded features and a 'price' column as the target.

    Returns:
    - GradientBoostingRegressor: The trained model.
    """
    # Separate features and target
    X = data.drop(columns=['price'])
    y = data['price']

    # Initialize and train the model
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X, y)

    return model

def tune_gradient_boosting(model: GradientBoostingRegressor, data: pd.DataFrame) -> GradientBoostingRegressor:
    """
    Tunes the hyperparameters of a GradientBoostingRegressor using GridSearchCV.

    Parameters:
    - data (pd.DataFrame): A DataFrame that includes pre-encoded features and a 'price' column as the target.

    Returns:
    - GradientBoostingRegressor: The best model after hyperparameter tuning.
    """
    # Separate features and target
    X = data.drop(columns=['price'])
    y = data['price']

    # Define the parameter grid (5 tested values for each hyperparameter)
    param_grid = {
        'n_estimators': [50, 100, 150, 200, 250],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3]
    }

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1
    )
    grid_search.fit(X, y)
    
    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Print the best parameters and performance
    print(f"Best Hyperparameters: {best_params}")
    print(f"Best Negative Mean Squared Error: {grid_search.best_score_:.2f}")

    return best_model
