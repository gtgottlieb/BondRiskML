import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from Roos import r2_oos
from bayesian_shrinkage import bayesian_shrinkage
from compute_benchmark import compute_benchmark_prediction
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV

def compute_benchmark_prediction(xr_insample, xr_oos):
    benchmark_preds = []

    for i in range(len(xr_oos)):
        combined = pd.concat([xr_insample, xr_oos.iloc[:i+1]]) 
        #avg_val = combined.mean()  # This computes column-wise means
        # This computes column-wise means from 12 months before
        avg_val = combined.iloc[:-12].mean() if len(combined) > 12 else combined.mean() 
        benchmark_preds.append(avg_val)

    # Convert list of Series (one per iteration) to a single DataFrames
    return pd.DataFrame(benchmark_preds, index=xr_oos.index)

# Reusable function that performs randomized grid search using the last 15% as validation.
def refit_rf_model(X, y):
    n_samples = X.shape[0]
    test_size = int(np.ceil(0.15 * n_samples))
    train_size = n_samples - test_size
    test_fold = np.concatenate((np.full(train_size, -1), np.zeros(test_size, dtype=int)))
    pre_split = PredefinedSplit(test_fold=test_fold)
    
    param_dist = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    rf = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        cv=pre_split,
        n_iter=10,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X, y)
    best_rf = random_search.best_estimator_
    best_rf.fit(X, y)
    return best_rf

def split_data_by_date(excess_returns: pd.DataFrame,
                       forward_rates: pd.DataFrame,
                       split_date, end_date,
                       macro_data: pd.DataFrame = None) -> dict:
    """
    Splits excess returns, forward rates, and optionally macro data into 
    in-sample and out-of-sample sets. The excess returns are shifted up by 1 year

    Args:
        excess_returns (pd.DataFrame): DataFrame with a 'Date' column.
        forward_rates (pd.DataFrame): DataFrame with a 'Date' column.
        split_date: Start date for the out-of-sample period.
        end_date: End date for the out-of-sample period.
        macro_data (pd.DataFrame, optional): DataFrame with a 'Date' column.

    Returns:
        dict: Dictionary containing in-sample and out-of-sample datasets.
    """

    # Ensure Date columns are datetime objects.
    excess_returns["Date"] = pd.to_datetime(excess_returns["Date"])
    forward_rates["Date"] = pd.to_datetime(forward_rates["Date"])
    if macro_data is not None:
        macro_data["Date"] = pd.to_datetime(macro_data["Date"])


    # Make sure that the excess returns are 1 year ahead of the forward rates and macro data.
    er_split_date = split_date + pd.DateOffset(months=12)
    er_end_date = end_date + pd.DateOffset(months=12)

  
    in_er = excess_returns.loc[excess_returns["Date"] < er_split_date].copy()
    out_er = excess_returns.loc[(excess_returns["Date"] >= er_split_date) &
                                (excess_returns["Date"] <= er_end_date)].copy()

    in_fr = forward_rates.loc[forward_rates["Date"] < split_date].copy()
    out_fr = forward_rates.loc[(forward_rates["Date"] >= split_date) &
                               (forward_rates["Date"] <= end_date)].copy()

    if macro_data is not None:
        in_macro = macro_data.loc[macro_data["Date"] < split_date].copy()
        out_macro = macro_data.loc[(macro_data["Date"] >= split_date) &
                                   (macro_data["Date"] <= end_date)].copy()
    else:
        in_macro, out_macro = None, None

    return {
        "excess_returns_in": in_er,
        "excess_returns_out": out_er,
        "forward_rates_in": in_fr,
        "forward_rates_out": out_fr,
        "macro_data_in": in_macro,
        "macro_data_out": out_macro,
    }


def iterative_rf_regression(er_in: pd.DataFrame,
                             fr_in: pd.DataFrame,
                             er_out: pd.DataFrame,
                             fr_out: pd.DataFrame,
                             macro_in: pd.DataFrame = None,
                             macro_out: pd.DataFrame = None,
                             n_macro_components: int = 8) -> pd.Series:
    """
    Performs iterative PCA regression with RandomForestRegressor and retrains the model
    using a reusable randomized grid search function (with last 15% split for CV).
    """
    predictions = []

    # Prepare IncrementalPCA for macro data if provided.
    if macro_in is not None:
        macro_scaler = MinMaxScaler(feature_range=(-1,1)).fit(macro_in)
        scaled_macro_in = macro_scaler.transform(macro_in)
        pca_macro = IncrementalPCA(n_components=n_macro_components)
        pca_macro.fit(scaled_macro_in)
        macro_pcs_in = pca_macro.transform(scaled_macro_in)
    else:
        macro_pcs_in = None


    # Combine forward and macro PCs as available.
    X_in = np.hstack([fr_in, macro_pcs_in]) if macro_pcs_in is not None else fr_in
    y_in = er_in.values.flatten()

    
    # Initial fit with randomized grid search using last 15% as validation.
    rf = refit_rf_model(X_in, y_in)

    # Iterate through out-of-sample observations.
    for idx in range(len(er_out)):
        # Transform current test sample for forward rates.
        fr_test = fr_out.iloc[[idx]]

        if macro_in is not None and macro_out is not None:
            macro_test = macro_out.iloc[[idx]]
            test_macro_scaled = macro_scaler.transform(macro_test)
            test_pcs_macro = pca_macro.transform(test_macro_scaled)
            X_test = np.hstack([fr_test, test_pcs_macro])
        else:
            X_test = fr_test

        # Predict the new observation.
        prediction = rf.predict(X_test)[0]
        predictions.append(prediction)

        # Append new observation into in-sample datasets using ignore_index.
        er_in = pd.concat([er_in, er_out.iloc[[idx]]], ignore_index=True)
        fr_in = pd.concat([fr_in, fr_out.iloc[[idx]]], ignore_index=True)
        if macro_in is not None and macro_out is not None:
            macro_in = pd.concat([macro_in, macro_out.iloc[[idx]]], ignore_index=True)
        

        if macro_in is not None:
            # Refit scaler and update IncrementalPCA for macro data.
            macro_scaler = MinMaxScaler(feature_range=(-1,1)).fit(macro_in)
            scaled_macro_in = macro_scaler.transform(macro_in)
            pca_macro.partial_fit(scaled_macro_in[-1:])  # partial update on the last row.
            macro_pcs_in = pca_macro.transform(scaled_macro_in)
            X_in = np.hstack([fr_in, macro_pcs_in])
        else:
            X_in = fr_in

        y_in = er_in.values.flatten()

        # Retrain model with updated in-sample data using cross validation.
        if idx >= 11:
            # Use delayed data (exclude the last 11 observations) to avoid data leakage.
            rf = refit_rf_model(X_in[:-11], y_in[:-11])


    return pd.Series(predictions, index=er_out.index)

def plot_oos_results(actual, predictions, benchmark, start_oos, end_oos, model=""):
    # Extract the oos dates from the xr.xlsx file.
    dates = pd.read_excel("data-folder/!Data for forecasting/xr.xlsx", usecols=["Date"])["Date"]
    dates = pd.to_datetime(dates)
    mask = (dates >= start_oos) & (dates <= end_oos)
    dates = dates.loc[mask].reset_index(drop=True)

    plt.figure(figsize=(10, 6))
    plt.plot(dates, actual.values, linestyle='--', label="Actual")
    plt.plot(dates, predictions.values, linestyle='-.', label=f"{model} Predictions")
    plt.plot(dates, benchmark.values, linestyle='-', label="Benchmark")
    plt.title("Out-of-Sample Comparison")
    plt.xlabel("Date")
    plt.ylabel("Return Values")
    plt.legend()
    plt.grid(True)
    plt.show()

def main(use_macro: bool, difference: bool = False):
    # Load datasets.
    forward_rates = pd.read_excel("data-folder/!Data for forecasting/forward_rates.xlsx")
    excess_returns = pd.read_excel("data-folder/!Data for forecasting/xr.xlsx")
    macro_data = pd.read_excel("data-folder/!Data for forecasting/Imputted_MacroData.xlsx") 

    # Difference the fwd rates to get stationary series.
    # Preserve the 'Date' column and compute differences for other columns.
    if difference:
        diff_fwd = forward_rates.iloc[:,1:] - forward_rates.iloc[:,1:].shift(12)
        diff_fwd = pd.concat([forward_rates["Date"], diff_fwd], axis=1)
        diff_fwd = diff_fwd.dropna()
        forward_rates = diff_fwd.copy()
        # Drop the first 12 observations of excess returns and macro data.
        excess_returns = excess_returns.iloc[12:].reset_index(drop=True)
        if macro_data is not None:
            macro_data = macro_data.iloc[12:].reset_index(drop=True)


    # Define out-of-sample period.
    start_oos = pd.to_datetime("1990-01-01")
    end_oos = pd.to_datetime("2023-11-01")

    # Convert 'Date' columns to datetime.
    for df in [forward_rates, excess_returns, macro_data]:
        df["Date"] = pd.to_datetime(df["Date"])
        
    # Use macro data only if flagged.
    macro_for_split = macro_data if use_macro else None

    # Split data into in-sample and out-of-sample.
    data_split = split_data_by_date(excess_returns, forward_rates, start_oos, end_oos, macro_data=macro_for_split)
    
    # Drop the 'Date' column.
    for key in data_split:
        if data_split[key] is not None:
            data_split[key] = data_split[key].drop(columns="Date")
    

    er_in = data_split["excess_returns_in"]
    er_out = data_split["excess_returns_out"]
    #realized = er_out.copy() # For computing IR
    #er_out.to_excel("data-folder/realized_xr.xlsx", index=False)
    fr_in = data_split["forward_rates_in"]
    fr_out = data_split["forward_rates_out"]
    macro_in = data_split["macro_data_in"]
    macro_out = data_split["macro_data_out"]

    # List of columns to predict.
    columns_to_predict = ["2 y", "3 y", "4 y", "5 y", "7 y", "10 y"]
    predictions = {}

    for col in columns_to_predict:
        print(f"Running iterative random forest regression for column: {col}")
        er_in_col = er_in[[col]].copy()
        er_out_col = er_out[[col]].copy()

        pred = iterative_rf_regression(
            er_in_col,
            fr_in.copy(),
            er_out_col,
            fr_out.copy(),
            macro_in=macro_in.copy() if macro_in is not None else None,
            macro_out=macro_out.copy() if macro_out is not None else None,
            n_macro_components=8  # Macro components are fixed at 8
        )
        predictions[col] = pred

    # Compute benchmark predictions.
    benchmark_preds = compute_benchmark_prediction(er_in, er_out)

    # Report out-of-sample R2 for each column.

    # Store all predictions  
    preds_df = pd.DataFrame()
    benchmark_df = pd.DataFrame()
    for col in predictions:

        # Save predictions
        preds_df[col] = predictions[col]
        benchmark_df[col] = benchmark_preds[col]

        

        # Plot the results using the reusable plotting function.
        plot_oos_results(er_out[col], predictions[col], benchmark_preds[col], start_oos, end_oos, model = "Rf")

        # Compute model Roos
        r2_value = r2_oos(er_out[col], predictions[col], benchmark_preds[col])
        print(f"Out-of-sample R2 for {col}: {r2_value}")

        bayes_preds = bayesian_shrinkage(benchmark_preds[col], predictions[col])
        r2_bayes = r2_oos(er_out[col], bayes_preds, benchmark_preds[col])
        print(f"Out-of-sample R2 with Bayesian shrinkage for {col}: {r2_bayes}")

  # Save to excel
    if difference:
        preds_df.to_excel("Extension code/Forecasting models/Saved preds/Random forest preds/diff_Macro_rf.xlsx", index=False)
    else:
        preds_df.to_excel("Extension code/Forecasting models/Saved preds/Random forest preds/Macro_rf.xlsx", index=False)

        
if __name__ == "__main__":
    # Directly call main with desired parameters.
    # Run it tomorrow!!
    main(use_macro=True, difference=True)
    main(use_macro=True, difference=False)

    