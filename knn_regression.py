import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_knn_regression(data):
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)

    y_train_pred = knn.predict(X_train)

    mse = mean_squared_error(y_train, y_train_pred)
    r2 = r2_score(y_train, y_train_pred)

    print(f"Training Mean Squared Error: {mse:.2f}")
    print(f"Training R^2 Score: {r2:.2f}")

    return knn

def validate_knn_model(model, data):
    """
    Validate the KNN model on a new dataset and tune hyperparameters.

    Args:
        model (KNeighborsRegressor): Trained KNN model.
        data (pd.DataFrame): Preprocessed and scaled dataframe for validation.

    Returns:
        KNeighborsRegressor: Best KNN model after hyperparameter tuning.
    """
    data = pd.get_dummies(data, columns=['cab_type'], drop_first=True)

    X = data.drop(columns=['price'])
    y = data['price']

    param_grid = {'n_neighbors': range(1, 21), 'weights': ['uniform', 'distance']}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    y_pred = best_model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"Best Hyperparameters: {best_params}")
    print(f"Validation Mean Squared Error: {mse:.2f}")
    print(f"Validation R^2 Score: {r2:.2f}")

    return best_model

def test_knn_model(model, data):
    """
    Test the KNN model to ensure no overfitting.

    Args:
        model (KNeighborsRegressor): Trained KNN model.
        data (pd.DataFrame): Preprocessed and scaled dataframe for testing.

    Returns:
        None
    """
    data = pd.get_dummies(data, columns=['cab_type'], drop_first=True)

    X = data.drop(columns=['price'])
    y = data['price']

    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"Test Mean Squared Error: {mse:.2f}")
    print(f"Test R^2 Score: {r2:.2f}")
