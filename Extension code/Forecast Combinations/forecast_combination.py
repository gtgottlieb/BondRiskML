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
    """
    Combine forecasts using simple average.
    
    Parameters:
        forecasts (list of np.ndarray]): List of forecast arrays from different models.
    Returns:
        np.ndarray: Combined forecast using simple average.
    """
    return np.mean(forecasts, axis=0)

def weighted_average_forecast(forecasts, weights):
    """
    Combine forecasts using weighted average.
    
    Parameters:
        forecasts (list of np.ndarray]): List of forecast arrays from different models.
        weights (list of float]): List of weights for each forecast.
    Returns:
        np.ndarray: Combined forecast using weighted average.
    """
    if len(forecasts) != len(weights):
        raise ValueError("Number of forecasts and weights must match.")
    return np.average(forecasts, axis=0, weights=weights)

###############################################################################
# 2. Helper function to load and split data once
###############################################################################
def load_and_split_data(
    start_oos=pd.to_datetime("1990-01-01"), 
    end_oos=pd.to_datetime("2018-12-01"),
    use_macro=False,
    macro_file="data-folder/!Data for forecasting/Imputted_MacroData.xlsx"
):
    """
    Loads the data, converts dates, and splits into in-sample and out-of-sample.
    Modify the file paths as needed.
    """
    # Load datasets (adjust your file paths as needed)
    forward_rates = pd.read_excel("data-folder/!Data for forecasting/forward_rates.xlsx")
    excess_returns = pd.read_excel("data-folder/!Data for forecasting/xr.xlsx")
    macro_data = pd.read_excel(macro_file)

    # Convert 'Date' columns
    forward_rates["Date"] = pd.to_datetime(forward_rates["Date"])
    excess_returns["Date"] = pd.to_datetime(excess_returns["Date"])
    macro_data["Date"] = pd.to_datetime(macro_data["Date"])

    # Optionally include macro data in the split
    macro_for_split = macro_data if use_macro else None

    # Split data
    data_split = split_data_by_date(
        excess_returns, 
        forward_rates, 
        start_oos, 
        end_oos, 
        macro_data=macro_for_split
    )

    # Drop 'Date' column in each subset (if present)
    for key in data_split:
        if data_split[key] is not None and "Date" in data_split[key].columns:
            data_split[key] = data_split[key].drop(columns="Date")

    # Return the dictionary containing everything
    return data_split

###############################################################################
# 3. Functions to run PCA and RF
###############################################################################
def run_pca_forecasts(data_split, n_fwd_components=3, n_macro_components=8):
    """
    Run PCA-based regression forecasts for selected maturities.
    Returns dict of predictions {maturity_col: np.array of forecasts}.
    """
    er_in = data_split["excess_returns_in"]
    er_out = data_split["excess_returns_out"]
    fr_in = data_split["forward_rates_in"]
    fr_out = data_split["forward_rates_out"]
    macro_in = data_split["macro_data_in"]
    macro_out = data_split["macro_data_out"]

    columns_to_predict = ["2 y", "3 y", "4 y", "5 y", "7 y", "10 y"]
    pca_preds = {}

    for col in columns_to_predict:
        print(f"Running PCA regression for column: {col}...")
        er_in_col = er_in[[col]]
        er_out_col = er_out[[col]]

        # iterative_pca_regression is presumably from pca_regression.py
        pred = iterative_pca_regression(
            er_in_col,
            fr_in,
            er_out_col,
            fr_out,
            macro_in=macro_in,
            macro_out=macro_out,
            n_fwd_components=n_fwd_components,
            n_macro_components=n_macro_components
        )
        pca_preds[col] = pred

    return pca_preds

def run_rf_forecasts(data_split, n_macro_components=8):
    """
    Run Random Forest-based regression forecasts for selected maturities.
    Returns dict of predictions {maturity_col: np.array of forecasts}.
    """
    er_in = data_split["excess_returns_in"]
    er_out = data_split["excess_returns_out"]
    fr_in = data_split["forward_rates_in"]
    fr_out = data_split["forward_rates_out"]
    macro_in = data_split["macro_data_in"]
    macro_out = data_split["macro_data_out"]

    columns_to_predict = ["2 y", "3 y", "4 y", "5 y", "7 y", "10 y"]
    rf_preds = {}

    for col in columns_to_predict:
        print(f"Running Random Forest regression for column: {col}...")
        er_in_col = er_in[[col]]
        er_out_col = er_out[[col]]

        # iterative_rf_regression presumably from random_forest.py
        pred = iterative_rf_regression(
            er_in_col,
            fr_in,
            er_out_col,
            fr_out,
            macro_in=macro_in,
            macro_out=macro_out,
            n_macro_components=n_macro_components
        )
        rf_preds[col] = pred

    return rf_preds

###############################################################################
# 4. Main forecasting function that runs PCA & RF, then combines forecasts
###############################################################################
def run_all_forecasts_and_combinations(
    start_oos=pd.to_datetime("1990-01-01"),
    end_oos=pd.to_datetime("2018-12-01"),
    n_fwd_components=3,
    use_macro=False,
    macro_file="data-folder/!Data for forecasting/Imputted_MacroData.xlsx"
):
    """
    1) Load and split data once.
    2) Run PCA and Random Forest forecasts.
    3) Compute benchmark predictions.
    4) Create various forecast combinations (simple average, weighted average).
    5) Apply Bayesian shrinkage (optional).
    6) Calculate out-of-sample R^2.
    
    Returns:
        dict: containing all forecast sets for each maturity.
    """
    # -----------------------
    # 1) Load data once
    # -----------------------
    data_split = load_and_split_data(
        start_oos=start_oos,
        end_oos=end_oos,
        use_macro=use_macro,
        macro_file=macro_file
    )
    er_in = data_split["excess_returns_in"]
    er_out = data_split["excess_returns_out"]

    # -----------------------
    # 2) Run forecasts
    # -----------------------
    pca_predictions = run_pca_forecasts(data_split, n_fwd_components=n_fwd_components)
    rf_predictions = run_rf_forecasts(data_split)
    
    # -----------------------
    # 3) Benchmark
    # -----------------------
    benchmark_predictions = compute_benchmark_prediction(er_in, er_out)

    # -----------------------
    # 4) Combine forecasts
    #    We'll combine [PCA, RF, Benchmark] for each maturity
    # -----------------------
    columns_to_predict = pca_predictions.keys()
    results_dict = {
        "PCA": {},
        "RF": {},
        "Benchmark": {},
        "SimpleAverage": {},
        "WeightedAverage": {},
        "BayesPCA": {},
        "BayesRF": {},
        "BayesAverage": {}
    }

    # Example weights: 40% PCA, 40% RF, 20% Benchmark
    w = [0.4, 0.4, 0.2]

    for col in columns_to_predict:
        # Retrieve forecasts for this maturity
        pca_col_pred = pca_predictions[col]
        rf_col_pred = rf_predictions[col]
        bench_col_pred = benchmark_predictions[col]
        actual_oos = er_out[col]

        # Store the raw forecasts
        results_dict["PCA"][col] = pca_col_pred
        results_dict["RF"][col] = rf_col_pred
        results_dict["Benchmark"][col] = bench_col_pred

        # Simple average
        combo_simple = simple_average_forecast([pca_col_pred, rf_col_pred, bench_col_pred])
        results_dict["SimpleAverage"][col] = combo_simple
        
        # Weighted average
        combo_weighted = weighted_average_forecast([pca_col_pred, rf_col_pred, bench_col_pred], w)
        results_dict["WeightedAverage"][col] = combo_weighted

        # -----------------------
        # 5) Apply Bayesian shrinkage to PCA, RF, and combination
        #    - You can also do Bayesian shrinkage on each separate forecast
        #      or on the combinations themselves.
        # -----------------------
        bayes_pca = bayesian_shrinkage(bench_col_pred, pca_col_pred)
        results_dict["BayesPCA"][col] = bayes_pca

        bayes_rf = bayesian_shrinkage(bench_col_pred, rf_col_pred)
        results_dict["BayesRF"][col] = bayes_rf

        # Bayesian shrinkage on the simple-average combination:
        bayes_avg = bayesian_shrinkage(bench_col_pred, combo_simple)
        results_dict["BayesAverage"][col] = bayes_avg

        # -----------------------
        # 6) Compute and print out-of-sample R² for each forecast
        #    (You can store them in a DataFrame if you like.)
        # -----------------------
        r2_pca = r2_oos(actual_oos, pca_col_pred, bench_col_pred)
        r2_rf  = r2_oos(actual_oos, rf_col_pred, bench_col_pred)
        r2_bench = r2_oos(actual_oos, bench_col_pred, bench_col_pred)  # Should be 0 by definition
        r2_simple = r2_oos(actual_oos, combo_simple, bench_col_pred)
        r2_weighted = r2_oos(actual_oos, combo_weighted, bench_col_pred)
        r2_bayes_pca = r2_oos(actual_oos, bayes_pca, bench_col_pred)
        r2_bayes_rf  = r2_oos(actual_oos, bayes_rf, bench_col_pred)
        r2_bayes_avg = r2_oos(actual_oos, bayes_avg, bench_col_pred)

        print(f"\n== {col} ==")
        print(f"R² (PCA)         : {r2_pca:.4f}")
        print(f"R² (RF)          : {r2_rf:.4f}")
        print(f"R² (Benchmark)   : {r2_bench:.4f}")  # usually 0
        print(f"R² (Simple Avg)  : {r2_simple:.4f}")
        print(f"R² (Weighted Avg): {r2_weighted:.4f}")
        print(f"R² (Bayes PCA)   : {r2_bayes_pca:.4f}")
        print(f"R² (Bayes RF)    : {r2_bayes_rf:.4f}")
        print(f"R² (Bayes Avg)   : {r2_bayes_avg:.4f}")

    return results_dict


if __name__ == "__main__":
    # Adjust n_fwd_components, use_macro, or macro_file path as needed
    results = run_all_forecasts_and_combinations(
        start_oos=pd.to_datetime("1990-01-01"),
        end_oos=pd.to_datetime("2018-12-01"),
        n_fwd_components=3,
        use_macro=False,  # Change to True if you want to incorporate macro_data
        macro_file="data-folder/!Data for forecasting/Imputted_MacroData.xlsx"
    )
    # results is a dict with all your forecasts and combos by maturity.
