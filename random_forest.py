import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def train_random_forest(data):
    data = pd.get_dummies(data, columns=['cab_type'], drop_first=True)

    X = data.drop(columns=['price'])
    y = data['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_train_pred = rf.predict(X_train)

    mse = mean_squared_error(y_train, y_train_pred)
    r2 = r2_score(y_train, y_train_pred)

    print(f"Random Forest - Training Mean Squared Error: {mse:.2f}")
    print(f"Random Forest - Training R^2 Score: {r2:.2f}")

    return rf

def validate_random_forest(model, data):
    data = pd.get_dummies(data, columns=['cab_type'], drop_first=True)

    X = data.drop(columns=['price'])
    y = data['price']

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    y_pred = best_model.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f" Random Forest - Best Hyperparameters: {best_params}")
    print(f"Random Forest - Validation Mean Squared Error: {mse:.2f}")
    print(f"Random Forest - Validation R^2 Score: {r2:.2f}")

    return best_model