import numpy as np
import pandas as pd

from pca_regression import split_data_by_date
from compute_benchmark import compute_benchmark_prediction
from Roos import r2_oos
from sklearn.linear_model import LinearRegression  # Meta-model for stacking
from sklearn.neural_network import MLPRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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

def linear_stacking(forecasts, actual):
    combined_forecast = []
    X = np.column_stack(forecasts)
    for i in range(len(actual)):
        if i <= 11:
            # Use equal weights for the first few predictions
            combined_forecast.append(np.mean(X[i, :]))
        else:
            # Train meta-model on past data only if sufficient data is available
            meta_model = LinearRegression()
            meta_model.fit(X[:i-11, :], actual[:i-11])
            # Predict for the current point
            combined_forecast.append(meta_model.predict(X[i, :].reshape(1, -1))[0])
    return np.array(combined_forecast)

def neural_network_stacking(forecasts, actual):
    """
    Combine forecasts using a neural network meta-model (stacked regression)
    with iterative training. For the first few observations (i <= 11),
    use equal weighting.
    
    Parameters:
        forecasts (list of np.ndarray): List of forecast arrays from different models.
        actual (np.ndarray): Actual observed values to train the meta-model.
    Returns:
        np.ndarray: Combined forecast using the neural network meta-model.
    """
    combined_forecast = []
    X = np.column_stack(forecasts)
    for i in range(len(actual)):
        if i <= 11:
            # Use equal weights for the first few predictions
            combined_forecast.append(np.mean(X[i, :]))
        else:
            # Train a simple neural network as meta-model using only past data
            nn_model = MLPRegressor(hidden_layer_sizes=(3,),
                                    activation='relu',
                                    solver='adam',
                                    random_state=42,
                                    max_iter=500)
            nn_model.fit(X[:i-11, :], actual[:i-11])
            # Predict for the current point
            combined_forecast.append(nn_model.predict(X[i, :].reshape(1, -1))[0])
    return np.array(combined_forecast)


if __name__ == "__main__":
    # Load data
    forward_rates = pd.read_excel("data-folder/!Data for forecasting/forward_rates.xlsx")
    excess_returns = pd.read_excel("data-folder/!Data for forecasting/xr.xlsx")
    macro_data = pd.read_excel("data-folder/!Data for forecasting/Imputted_MacroData.xlsx") 

    # Define out-of-sample period.
    start_oos = pd.to_datetime("1990-01-01")
    end_oos = pd.to_datetime("2023-12-01")

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

    # Instead of hardcoding two predictions, define a list of prediction file paths.
    prediction_files = [
        "Extension code/Forecast Combinations/Predictions/PCA/FWD_reg.xlsx",
        "Extension code/Forecast Combinations/Predictions/RF/FWD_rf.xlsx",
        "Extension code/Forecast Combinations/Predictions/ElasticNet/FWD_en.xlsx",
    ]
    predictions_list = [pd.read_excel(f) for f in prediction_files]
    
    # Calculate benchmark predictions using in-sample and out-of-sample excess returns
    benchmark_preds = compute_benchmark_prediction(er_in, er_out)

    # Define burn-in period (approx 10 years for monthly data)
    burn_in = 120

    # Loop over each maturity column (use first predictions dataframe columns)
    for col in predictions_list[0].columns:
        # Calculate OOS R² for each individual prediction input after burn-in.])
        for idx, pred_df in enumerate(predictions_list):
            r2_val = r2_oos(er_out[col].values[burn_in:], pred_df[col].values[burn_in:], benchmark_preds[col].values[burn_in:])
            print(f"Column {col} - Prediction {idx+1} OOS R²: {r2_val:.4f}")

        # Combine forecasts from all prediction sources.
        # Create a list of forecast arrays (each prediction column as a numpy array).
        forecasts = [pred_df[col].values for pred_df in predictions_list]
        simple_combo = simple_average_forecast(forecasts)
        stacked_combo = linear_stacking(forecasts, er_out[col].values)
        neural_combo = neural_network_stacking(forecasts, er_out[col].values)
        
        # Compute combined OOS R² scores after burn-in.
        r2_simple = r2_oos(er_out[col].values[burn_in:], simple_combo[burn_in:], benchmark_preds[col].values[burn_in:])
        r2_linear = r2_oos(er_out[col].values[burn_in:], stacked_combo[burn_in:], benchmark_preds[col].values[burn_in:])
        r2_neural = r2_oos(er_out[col].values[burn_in:], neural_combo[burn_in:], benchmark_preds[col].values[burn_in:])

        print(f"Column {col} - Combined Avg R²: {r2_simple:.4f}, Linear R²: {r2_linear:.4f}, NN R²: {r2_neural:.4f}")

