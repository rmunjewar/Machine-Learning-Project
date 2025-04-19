import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def run_evaluation(models, test_data):
    print("\nEVALUATION RESULTS:")
    print("-" * 50)
    
    # Results dictionary
    results = {
        'Model': [],
        'MAE': [],
        'MSE': [],
        'RMSE': [],
        'R²': []
    }
    
    # Evaluate each model
    for name, model in models.items():
        try:
            if name in ['Random Forest', 'Linear Regression']:
                # Create dummy variables like in training
                test_encoded = pd.get_dummies(test_data, columns=['cab_type'], drop_first=True)
                X_test = test_encoded.drop(columns=['price'])
            else:
                # Use raw data for other models
                X_test = test_data.drop(columns=['price'])
                
            y_test = test_data['price']
            
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            results['Model'].append(name)
            results['MAE'].append(round(mae, 3))
            results['MSE'].append(round(mse, 3))
            results['RMSE'].append(round(rmse, 3))
            results['R²'].append(round(r2, 3))
            
            # Print individual model results
            print(f"\n{name} Model:")
            print(f"  MAE:  {mae:.3f}")
            print(f"  MSE:  {mse:.3f}")
            print(f"  RMSE: {rmse:.3f}")
            print(f"  R²:   {r2:.3f}")
                
        except Exception as model_error:
            print(f"Error evaluating {name} model: {model_error}")
    

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('MAE')
  
    print("\nModel Comparison:")
    print(results_df)
    
    if len(results_df) > 0:
        best_model = results_df.iloc[0]['Model']
        print(f"\nBest Model: {best_model}")
        print(f"MAE: {results_df.iloc[0]['MAE']}")
        return best_model
    else:
        print("No models were successfully evaluated")
        return None
