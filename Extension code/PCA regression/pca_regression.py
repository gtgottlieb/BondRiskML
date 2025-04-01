import argparse
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from compute_benchmark import compute_benchmark_prediction
from Roos import r2_oos

def split_data_by_date(excess_returns: pd.DataFrame,
                       forward_rates: pd.DataFrame,
                       split_date, end_date,
                       macro_data: pd.DataFrame = None) -> dict:
    """
    Splits excess returns, forward rates, and optionally macro data into 
    in-sample and out-of-sample sets.

    Args:
        excess_returns (pd.DataFrame): DataFrame with a 'Date' column.
        forward_rates (pd.DataFrame): DataFrame with a 'Date' column.
        split_date: Start date for the out-of-sample period.
        end_date: End date for the out-of-sample period.
        macro_data (pd.DataFrame, optional): DataFrame with a 'Date' column.

    Returns:
        dict: Dictionary containing in-sample and out-of-sample datasets.
    """
    in_er = excess_returns.loc[excess_returns["Date"] < split_date].copy()
    out_er = excess_returns.loc[(excess_returns["Date"] >= split_date) & 
                                (excess_returns["Date"] <= end_date)].copy()

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

def iterative_pca_regression(er_in: pd.DataFrame,
                             fr_in: pd.DataFrame,
                             er_out: pd.DataFrame,
                             fr_out: pd.DataFrame,
                             macro_in: pd.DataFrame = None,
                             macro_out: pd.DataFrame = None,
                             n_fwd_components: int = 3,
                             n_macro_components: int = 8) -> pd.Series:
    """
    Performs iterative PCA regression. At each step the model is trained on the 
    current in-sample data, predicts the next out-of-sample observation, then the 
    new data point is added to the training set.

    Args:
        er_in (pd.DataFrame): In-sample excess returns.
        fr_in (pd.DataFrame): In-sample forward rates.
        er_out (pd.DataFrame): Out-of-sample excess returns.
        fr_out (pd.DataFrame): Out-of-sample forward rates.
        macro_in (pd.DataFrame, optional): In-sample macro data.
        macro_out (pd.DataFrame, optional): Out-of-sample macro data.
        n_fwd_components (int): Number of PCA components for forward rates.
        n_macro_components (int): Number of PCA components for macro data (fixed at 8).

    Returns:
        pd.Series: Predictions for out-of-sample observations.
    """
    predictions = []

    # Prepare PCA for macro data if provided.
    if macro_in is not None:
        macro_scaler = StandardScaler().fit(macro_in)
        scaled_macro_in = macro_scaler.transform(macro_in)
        pca_macro = PCA(n_components=n_macro_components).fit(scaled_macro_in)
        macro_pcs_in = pca_macro.transform(scaled_macro_in)
    else:
        macro_pcs_in = None

    # PCA for forward rates.
    pca_fwd = PCA(n_components=n_fwd_components).fit(fr_in)
    pcs_fwd_in = pca_fwd.transform(fr_in)

    # Combine forward and macro PCs as available.
    X_in = np.hstack([pcs_fwd_in, macro_pcs_in]) if macro_pcs_in is not None else pcs_fwd_in
    y_in = er_in.values
    model = LinearRegression().fit(X_in, y_in)

    # Iterate through out-of-sample observations.
    for idx in range(len(er_out)):
        # Transform current test sample for forward rates.
        fr_test = fr_out.iloc[[idx]]
        test_pcs_fwd = pca_fwd.transform(fr_test)

        if macro_in is not None and macro_out is not None:
            macro_test = macro_out.iloc[[idx]]
            test_macro_scaled = macro_scaler.transform(macro_test)
            test_pcs_macro = pca_macro.transform(test_macro_scaled)
            X_test = np.hstack([test_pcs_fwd, test_pcs_macro])
        else:
            X_test = test_pcs_fwd

        # Predict the new observation.
        prediction = model.predict(X_test)[0][0]
        predictions.append(prediction)

        # Append new observation into in-sample datasets.
        er_in = pd.concat([er_in, er_out.iloc[[idx]]])
        fr_in = pd.concat([fr_in, fr_out.iloc[[idx]]])
        if macro_in is not None and macro_out is not None:
            macro_in = pd.concat([macro_in, macro_out.iloc[[idx]]])

        # Refit PCA and regression model with the updated in-sample data.
        pca_fwd = PCA(n_components=n_fwd_components).fit(fr_in)
        pcs_fwd_in = pca_fwd.transform(fr_in)

        if macro_in is not None:
            macro_scaler = StandardScaler().fit(macro_in)
            scaled_macro_in = macro_scaler.transform(macro_in)
            pca_macro = PCA(n_components=n_macro_components).fit(scaled_macro_in)
            macro_pcs_in = pca_macro.transform(scaled_macro_in)
            X_in = np.hstack([pcs_fwd_in, macro_pcs_in])
        else:
            X_in = pcs_fwd_in

        y_in = er_in.values
        model.fit(X_in, y_in)

    return pd.Series(predictions, index=er_out.index)

def main(n_fwd_components: int, use_macro: bool):
    # Load datasets.
    forward_rates = pd.read_excel("data-folder/Fwd rates and xr/forward_rates.xlsx")
    excess_returns = pd.read_excel("data-folder/Fwd rates and xr/xr.xlsx")
    macro_data = pd.read_excel("data-folder/Cleaned data/Yields+Final/Imputted_MacroData.xlsx")  # Example path

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
        print(f"Running iterative PCA regression for column: {col}")
        er_in_col = er_in[[col]].copy()
        er_out_col = er_out[[col]].copy()

        pred = iterative_pca_regression(
            er_in_col,
            fr_in.copy(),
            er_out_col,
            fr_out.copy(),
            macro_in=macro_in.copy() if macro_in is not None else None,
            macro_out=macro_out.copy() if macro_out is not None else None,
            n_fwd_components=n_fwd_components,
            n_macro_components=8  # Macro components are fixed at 8
        )
        predictions[col] = pred

    # Compute benchmark predictions.
    benchmark_preds = compute_benchmark_prediction(er_in, er_out)

    # Report out-of-sample R2 for each column.
    for col in predictions:
        r2_value = r2_oos(er_out[col], predictions[col], benchmark_preds[col])
        print(f"Out-of-sample R2 for {col}: {r2_value}")

if __name__ == "__main__":
    # Directly call main with desired parameters.
    main(n_fwd_components=5, use_macro=False)