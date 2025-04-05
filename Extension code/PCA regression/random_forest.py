import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from compute_benchmark import compute_benchmark_prediction
from Roos import r2_oos
from bayesian_shrinkage import bayesian_shrinkage
from splitting_data import split_data_by_date
import matplotlib.pyplot as plt

def iterative_rf_regression(er_in: pd.DataFrame,
                            fr_in: pd.DataFrame,
                            er_out: pd.DataFrame,
                            fr_out: pd.DataFrame,
                            macro_in: pd.DataFrame = None,
                            macro_out: pd.DataFrame = None) -> pd.Series:
    """
    Performs iterative random forest regression.
    At each iteration, the model is trained on the in-sample data using the first 10 forward rates as features,
    then one prediction is generated for the current out‑of‑sample observation.
    """
    predictions = []

    for idx in range(len(er_out)):
        X_train = fr_in
        y_train = er_in
        
        # Initialize and train the random forest model.
        rf = RandomForestRegressor(random_state=42)
        rf.fit(X_train, y_train)
        
        # Select the corresponding test observation
        X_test = fr_out.iloc[[idx]]
        pred = rf.predict(X_test)[0]
        predictions.append(pred)
        
        # Append the new observation into the in-sample datasets.
        er_in = pd.concat([er_in, er_out.iloc[[idx]]], ignore_index=True)
        fr_in = pd.concat([fr_in, fr_out.iloc[[idx]]], ignore_index=True)

    return pd.Series(predictions, index=er_out.index)

def main(n_fwd_components: int, use_macro: bool):
    # Load datasets.
    forward_rates = pd.read_excel("data-folder/Fwd rates and xr/forward_rates.xlsx")
    excess_returns = pd.read_excel("data-folder/Fwd rates and xr/xr.xlsx")
    macro_data = pd.read_excel("data-folder/Cleaned data/Yields+Final/Imputted_MacroData.xlsx") 
    
    

    # Define out-of-sample period.
    start_oos = "1990-01-01"
    end_oos = "2018-12-01"

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
    
    # If using macro data, align the datasets by truncating to the minimum length.
    if use_macro and data_split["macro_data_in"] is not None:
        min_in = min(len(data_split["excess_returns_in"]),
                     len(data_split["forward_rates_in"]),
                     len(data_split["macro_data_in"]))
        data_split["excess_returns_in"] = data_split["excess_returns_in"].iloc[:min_in]
        data_split["forward_rates_in"] = data_split["forward_rates_in"].iloc[:min_in]
        data_split["macro_data_in"] = data_split["macro_data_in"].iloc[:min_in]
    if use_macro and data_split["macro_data_out"] is not None:
        min_out = min(len(data_split["excess_returns_out"]),
                      len(data_split["forward_rates_out"]),
                      len(data_split["macro_data_out"]))
        data_split["excess_returns_out"] = data_split["excess_returns_out"].iloc[:min_out]
        data_split["forward_rates_out"] = data_split["forward_rates_out"].iloc[:min_out]
        data_split["macro_data_out"] = data_split["macro_data_out"].iloc[:min_out]

    er_in = data_split["excess_returns_in"]
    er_out = data_split["excess_returns_out"]
    fr_in = data_split["forward_rates_in"]
    fr_out = data_split["forward_rates_out"]
    macro_in = data_split["macro_data_in"]
    macro_out = data_split["macro_data_out"]

    # List of columns to predict.
    columns_to_predict = ["2 y", "3 y", "4 y", "5 y", "7 y", "10 y"]
    predictions = {}

    for col in columns_to_predict:
        print(f"Running iterative RF regression for column: {col}")
        er_in_col = er_in[[col]].copy()
        er_out_col = er_out[[col]].copy()

        pred = iterative_rf_regression(
            er_in_col,
            fr_in.copy(),
            er_out_col,
            fr_out.copy(),
            macro_in=macro_in.copy() if macro_in is not None else None,
            macro_out=macro_out.copy() if macro_out is not None else None
        )
        predictions[col] = pred

    # Compute benchmark predictions.
    benchmark_preds = compute_benchmark_prediction(er_in, er_out)

    # Report out-of-sample R2 for each column.
    for col in predictions:

        '''
        # Extract the oos date
        dates = pd.read_excel("data-folder/Fwd rates and xr/xr.xlsx", usecols=["Date"])["Date"]
        dates = pd.to_datetime(dates)
        mask = (dates >= pd.to_datetime("1990-01-01")) & (dates <= pd.to_datetime("2018-12-01"))
        dates = dates.loc[mask].reset_index(drop=True)

        # Plot the predictions vs benchmark vs actuals for each column.
        plt.figure(figsize=(10, 6))
        plt.plot(dates, er_out[col].values, linestyle='--', label="Actual")
        plt.plot(dates, predictions[col].values, linestyle='-.', label="RF Predictions")
        plt.plot(dates, benchmark_preds[col].values, linestyle='-', label="Benchmark")
        plt.title(f"Out-of-Sample Comparison for {col}")
        plt.xlabel("Date")
        plt.ylabel("Return Values")
        plt.legend()
        plt.grid(True)
        plt.show()
        '''
        
        # Compute model Roos
        r2_value = r2_oos(er_out[col], predictions[col], benchmark_preds[col])
        print(f"Out-of-sample R2 for {col}: {r2_value}")

        # Compute model Roos with Bayesian shrinkage
        bayes_preds = bayesian_shrinkage(benchmark_preds[col], predictions[col])
        r2_bayes = r2_oos(er_out[col], bayes_preds, benchmark_preds[col])
        print(f"Out-of-sample R2 with Bayesian shrinkage for {col}: {r2_bayes}")
        
if __name__ == "__main__":
    # Directly call main with desired parameters.
    main(n_fwd_components=10, use_macro=False)