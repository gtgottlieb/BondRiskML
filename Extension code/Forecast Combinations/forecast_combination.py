import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Add this import for plotting

from pca_regression import split_data_by_date
from compute_benchmark import compute_benchmark_prediction
from Roos import r2_oos
from Roos import get_CW_adjusted_R2_signif as RSZ_Signif
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
    combined_forecast = []
    X = np.column_stack(forecasts)
    for i in range(len(actual)):
        if i <= 11:
            # Use equal weights for the first few predictions
            combined_forecast.append(np.mean(X[i, :]))
        else:
            # Train a simple neural network as meta-model using only past data
            nn_model = MLPRegressor(hidden_layer_sizes=(3),
                                    activation='relu',
                                    solver='adam',
                                    random_state=42,
                                    max_iter=500)
            nn_model.fit(X[:i-11, :], actual[:i-11])
            # Predict for the current point
            combined_forecast.append(nn_model.predict(X[i, :].reshape(1, -1))[0])
    return np.array(combined_forecast)


def main(predictions_list, output_file):
    # Load data
    forward_rates = pd.read_excel("data-folder/!Data for forecasting/forward_rates.xlsx")
    excess_returns = pd.read_excel("data-folder/!Data for forecasting/xr.xlsx")
    macro_data = pd.read_excel("data-folder/!Data for forecasting/Imputted_MacroData.xlsx") 

    # Define out-of-sample period.
    start_oos = pd.to_datetime("1991-01-01")
    end_oos = pd.to_datetime("2023-11-01")

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

    # Calculate benchmark predictions using in-sample and out-of-sample excess returns
    benchmark_preds = compute_benchmark_prediction(er_in, er_out)

    # Define burn-in period (approx 10 years for monthly data)
    burn_in = 120

    # Loop over each maturity column (use first predictions dataframe columns)
    neural_results = {}
    for col in predictions_list[0].columns:
        # Calculate OOS R² for each individual prediction input after burn-in.
        for idx, pred_df in enumerate(predictions_list):
            r2_val = r2_oos(er_out[col].values, pred_df[col].values, benchmark_preds[col].values)
            tstat_val, p_value = RSZ_Signif(er_out[col].values, pred_df[col].values, benchmark_preds[col].values)
            print(f"Col {col} - Pred {idx+1}: R²={r2_val:.4f}, t-stat={tstat_val:.4f}, p-val={p_value:.4f}")

        # Combine forecasts from all prediction sources.
        # Create a list of forecast arrays (each prediction column as a numpy array).
        forecasts = [pred_df[col].values for pred_df in predictions_list]
        simple_combo = simple_average_forecast(forecasts)
        stacked_combo = linear_stacking(forecasts, er_out[col].values)
        neural_combo = neural_network_stacking(forecasts, er_out[col].values)
        
        # Store neural network stacking results for saving later.
        neural_results[col] = neural_combo

        # Compute combined OOS R² scores and significance levels after burn-in.
        r2_simple = r2_oos(er_out[col].values[burn_in:], simple_combo[burn_in:], benchmark_preds[col].values[burn_in:])
        tstat_simple, pval_simple = RSZ_Signif(er_out[col].values[burn_in:], simple_combo[burn_in:], benchmark_preds[col].values[burn_in:])
        r2_linear = r2_oos(er_out[col].values[burn_in:], stacked_combo[burn_in:], benchmark_preds[col].values[burn_in:])
        tstat_linear, pval_linear = RSZ_Signif(er_out[col].values[burn_in:], stacked_combo[burn_in:], benchmark_preds[col].values[burn_in:])
        r2_neural = r2_oos(er_out[col].values[burn_in:], neural_combo[burn_in:], benchmark_preds[col].values[burn_in:])
        tstat_neural, pval_neural = RSZ_Signif(er_out[col].values[burn_in:], neural_combo[burn_in:], benchmark_preds[col].values[burn_in:])

        print(f"Col {col} - Simple: R²={r2_simple:.4f}, t-stat={tstat_simple:.4f}, p-val={pval_simple:.4f} | "
              f"Linear: R²={r2_linear:.4f}, t-stat={tstat_linear:.4f}, p-val={pval_linear:.4f} | "
              f"NN: R²={r2_neural:.4f}, t-stat={tstat_neural:.4f}, p-val={pval_neural:.4f}")
        
        # Plot the neural network stacking results for each maturity.
        plt.figure(figsize=(10, 6))
        plt.plot(neural_combo, label="Neural Network Stacking", color="blue")
        plt.plot(er_out[col].values, label="Actual", color="orange", linestyle="--")
        plt.title(f"Forecast Combination for Maturity: {col}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        output_filename = output_file.split("/")[-1].rsplit(".", 1)[0]  # Extract only the file name without extension
        plt.savefig(f"Extension code/Forecast Combinations/Plots/{output_filename}_{col}.png")
        plt.close()

    # Save neural network stacking results to an Excel file.
    neural_df = pd.DataFrame(neural_results)
    neural_df.columns = predictions_list[0].columns  # Ensure proper column names
    neural_df.to_excel(output_file, index=False)


if __name__ == "__main__":
    # Example usage with dataframes
    pca_fwd = pd.read_excel("Extension code/Forecast Combinations/Predictions/PCA/FWD_reg.xlsx")
    rf_fwd = pd.read_excel("Extension code/Forecast Combinations/Predictions/RF/FWD_rf.xlsx")
    en_fwd = pd.read_excel("Extension code/Forecast Combinations/Predictions/ElasticNet/FWD_en.xlsx")
    nn_fwd = pd.read_excel("Extension code/Forecast Combinations/Predictions/NN/FWD_nn.xlsx")

    pca_fwd_diff = pd.read_excel("Extension code/Forecast Combinations/Predictions/PCA/diff_FWD.xlsx")
    rf_fwd_diff = pd.read_excel("Extension code/Forecast Combinations/Predictions/RF/diff_FWD_rf.xlsx")
    en_fwd_diff = pd.read_excel("Extension code/Forecast Combinations/Predictions/ElasticNet/diff_FWD_en.xlsx")
    

    pca_macro = pd.read_excel("Extension code/Forecast Combinations/Predictions/PCA/Macro_reg.xlsx")
    rf_macro = pd.read_excel("Extension code/Forecast Combinations/Predictions/RF/Macro_rf.xlsx")
    en_macro = pd.read_excel("Extension code/Forecast Combinations/Predictions/ElasticNet/Macro_en.xlsx")
    nn_macro = pd.read_excel("Extension code/Forecast Combinations/Predictions/NN/NN3_32_16_8_Predictions.xlsx")

    pca_macro_diff = pd.read_excel("Extension code/Forecast Combinations/Predictions/PCA/diff_Macro.xlsx")
    rf_macro_diff = pd.read_excel("Extension code/Forecast Combinations/Predictions/RF/diff_Macro_rf.xlsx")
    en_macro_diff = pd.read_excel("Extension code/Forecast Combinations/Predictions/ElasticNet/diff_Macro_en.xlsx")
    nn_macro_diff = pd.read_excel("Extension code/Forecast Combinations/Predictions/NN/NN3_32_16_8_diff.xlsx")

    predictions_list = [pca_macro[12:], nn_macro]
    output_file = "Extension code/Forecast Combinations/Combo Predictions/PCA_NeuralNet_Macro.xlsx"
    print(f"PCA + NN Macro")
    main(predictions_list, output_file)

    predictions_list = [rf_macro[12:], nn_macro]
    output_file = "Extension code/Forecast Combinations/Combo Predictions/RF_NeuralNet_FWD.xlsx"
    print(f"PCA + NN Macro")
    main(predictions_list, output_file)

    predictions_list = [en_macro[12:], nn_macro]
    output_file = "Extension code/Forecast Combinations/Combo Predictions/RF_NeuralNet_FWD.xlsx"
    print(f"PCA + NN Macro")
    main(predictions_list, output_file)

    
    """
    # Macro
    print("Macro:")
    prediction_files = [
        "Extension code/Forecast Combinations/Predictions/PCA/Macro_reg.xlsx",
        "Extension code/Forecast Combinations/Predictions/ElasticNet/Macro_en.xlsx"
    ]
    
    output_file = "Extension code/Forecast Combinations/Combo Predictions/PCA_ElasticNet_Macro.xlsx"
    
    main(prediction_files, output_file)

    # Difference FWD
    print("Difference FWD:")
    prediction_files = [
        "Extension code/Forecast Combinations/Predictions/PCA/diff_FWD_reg.xlsx",
        "Extension code/Forecast Combinations/Predictions/ElasticNet/diff_FWD_en.xlsx"
    ]
    
    output_file = "Extension code/Forecast Combinations/Combo Predictions/PCA_ElasticNet_FWD_DIFF.xlsx"
    
    main(prediction_files, output_file)

    # Difference Macro
    print("Difference Macro:")
    prediction_files = [
        "Extension code/Forecast Combinations/Predictions/PCA/diff_Macro_reg.xlsx",
        "Extension code/Forecast Combinations/Predictions/ElasticNet/diff_Macro_en.xlsx"
    ]
    
    output_file = "Extension code/Forecast Combinations/Combo Predictions/PCA_ElasticNet_Macro_DIFF.xlsx" 

    main(prediction_files, output_file)
    """