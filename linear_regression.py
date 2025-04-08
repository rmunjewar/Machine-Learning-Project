import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_linear_regression(data):
    data = pd.get_dummies(data, columns=['cab_type'], drop_first=True)

    X = data.drop(columns=['price'])
    y = data['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    mse = mean_squared_error(y_train, y_train_pred)
    r2 = r2_score(y_train, y_train_pred)

    print(f"Linear Regression - Training MSE: {mse:.2f}")
    print(f"Linear Regression - R^2: {r2:.2f}")

    return model

# def test_linear_regression(model, data):
#     data = pd.get_dummies(data, columns=['cab_type'], drop_first=True)

#     X = data.drop(columns=['price'])
#     y = data['price']

#     y_pred = model.predict(X)

#     mse = mean_squared_error(y, y_pred)
#     r2 = r2_score(y, y_pred)

#     pprint(f"Linear Regression - Training MSE: {mse:.2f}")
#     print(f"Linear Regression - R^2: {r2:.2f}")
