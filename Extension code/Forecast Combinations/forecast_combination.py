import numpy as np
import pandas as pd

from pca_regression import iterative_pca_regression, split_data_by_date
from compute_benchmark import compute_benchmark_prediction
from Roos import r2_oos
from bayesian_shrinkage import bayesian_shrinkage
from random_forest import iterative_rf_regression  # Ensure this function is defined or imported

###############################################################################
# 1. Forecast Combination Methods
###############################################################################
def simple_average_forecast(forecasts):

    return np.mean(forecasts, axis=0)

def weighted_average_forecast(forecasts, actual):
    
    mse_list = []
    for i, forecast in enumerate(forecasts):
        mse = np.mean((actual - forecast) ** 2)
        mse_list.append(mse)
    mse_array = np.array(mse_list)
    # Prevent division by zero.
    mse_array[mse_array == 0] = 1e-10
    weights = 1.0 / mse_array
    
    weights /= np.sum(weights)
    combined = np.zeros_like(forecasts[0])
    for w, forecast in zip(weights, forecasts):
        combined += w * forecast
    return combined


if __name__ == "__main__":
    # Load data
    forward_rates = pd.read_excel("data-folder/!Data for forecasting/forward_rates.xlsx")
    excess_returns = pd.read_excel("data-folder/!Data for forecasting/xr.xlsx")
    macro_data = pd.read_excel("data-folder/!Data for forecasting/Imputted_MacroData.xlsx") 

    # Define out-of-sample period.
    start_oos = pd.to_datetime("1990-01-01")
    end_oos = pd.to_datetime("2018-12-01")

    # Convert 'Date' columns to datetime.
    for df in [forward_rates, excess_returns, macro_data]:
        df["Date"] = pd.to_datetime(df["Date"])
        
    # Split data into in-sample and out-of-sample.
    data_split = split_data_by_date(excess_returns, forward_rates, start_oos, end_oos, macro_data=None)
    
    # Drop the 'Date' column.
    for key in data_split:
        if data_split[key] is not None:
            data_split[key] = data_split[key].drop(columns="Date")
    

    er_in = data_split["excess_returns_in"]
    er_out = data_split["excess_returns_out"]
    realized = er_out.copy() # For computing IR
    #er_out.to_excel("data-folder/realized_xr.xlsx", index=False)
    fr_in = data_split["forward_rates_in"]
    fr_out = data_split["forward_rates_out"]

    # Load predictions (as excel sheets or call the function)
    predictions_1 = pd.read_excel("Extension code/Forecast Combinations/Predictions/PCA/PCA_fwd.xlsx")
    predictions_2 = pd.read_excel("Extension code/Forecast Combinations/Predictions/Random forest/FWD only/FWD_rf.xlsx")

    # Calculate benchmark predictions using in-sample and out-of-sample excess returns
    benchmark_preds = compute_benchmark_prediction(er_in, er_out)
    
    # Calculate and print OOS R² scores for each maturity column for both prediction sets")
    for col in predictions_1.columns:
        r2_pca = r2_oos(er_out[col].values, predictions_1[col].values, benchmark_preds[col].values)
        r2_rf  = r2_oos(er_out[col].values, predictions_2[col].values, benchmark_preds[col].values)
        print(f"Column {col} - PCA OOS R²: {r2_pca:.4f}, RF OOS R²: {r2_rf:.4f}")
        
        # Combine forecasts from PCA and RF using simple and weighted average methods.
        forecasts = [predictions_1[col].values, predictions_2[col].values]
        simple_combo = simple_average_forecast(forecasts)
        weighted_combo = weighted_average_forecast(forecasts, er_out[col].values)
        
        # Compute OOS R² scores for the combined forecasts.
        r2_simple = r2_oos(er_out[col].values, simple_combo, benchmark_preds[col].values)
        r2_weighted = r2_oos(er_out[col].values, weighted_combo, benchmark_preds[col].values)
        
        print(f"Column {col} - Simple Average Combo OOS R²: {r2_simple:.4f}, Weighted Average Combo OOS R²: {r2_weighted:.4f}")

