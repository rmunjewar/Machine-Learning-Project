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
    results_df['MAE_rank'] = results_df['MAE'].rank()
    results_df['MSE_rank'] = results_df['MSE'].rank()
    results_df['RMSE_rank'] = results_df['RMSE'].rank()
    results_df['R²_rank'] = results_df['R²'].rank(ascending=False)  # For R², higher is better
    
    # Calculate average rank across all metrics
    results_df['Avg_Rank'] = (results_df['MAE_rank'] + 
                              results_df['MSE_rank'] + 
                              results_df['RMSE_rank'] + 
                              results_df['R²_rank']) / 4
    
    results_df = results_df.sort_values('Avg_Rank')
    
    display_df = results_df.drop(columns=['MAE_rank', 'MSE_rank', 'RMSE_rank', 'R²_rank', 'Avg_Rank'])
  
    print("\nModel Comparison:")
    print(display_df.to_string(index=False))
    
    print("\nDetailed Ranking:")
    ranking_df = results_df[['Model', 'MAE_rank', 'MSE_rank', 'RMSE_rank', 'R²_rank', 'Avg_Rank']]
    ranking_df = ranking_df.sort_values('Avg_Rank')
    print(ranking_df.to_string(index=False))
    
    if len(results_df) > 0:
        best_model = results_df.iloc[0]['Model']
        print(f"\nBest Model Overall: {best_model}")
        print(f"  MAE:  {results_df.iloc[0]['MAE']}")
        print(f"  MSE:  {results_df.iloc[0]['MSE']}")
        print(f"  RMSE: {results_df.iloc[0]['RMSE']}")
        print(f"  R²:   {results_df.iloc[0]['R²']}")
        return best_model
    else:
        print("No models were successfully evaluated")
        return None
