from sklearn.decomposition import PCA
import pandas as pd
from sklearn.linear_model import LinearRegression
from compute_benchmark import compute_benchmark_prediction
from Roos import r2_oos
from sklearn.preprocessing import StandardScaler
import numpy as np

def split_data_by_date(excess_returns, forward_rates, split_date, end_date, Macro_data=None):
    """
    Splits excess_returns and forward_rates into in-sample and out-of-sample datasets based on a predefined date range.

    Args:
        excess_returns (pd.DataFrame): DataFrame containing excess returns with a 'Date' column.
        forward_rates (pd.DataFrame): DataFrame containing forward rates with a 'Date' column.
        split_date (str): The start date for the out-of-sample period, in "YYYY-MM-DD" format.
        end_date (str): The end date for the out-of-sample period, in "YYYY-MM-DD" format.

    Returns:
        dict: A dictionary containing in-sample and out-of-sample data for both excess_returns and forward_rates.
    """

    # Split excess_returns
    excess_returns_insample = excess_returns.loc[excess_returns["Date"] < split_date]
    excess_returns_oos = excess_returns.loc[
        (excess_returns["Date"] >= split_date) & (excess_returns["Date"] <= end_date)
    ]

    # Split forward_rates
    forward_rates_insample = forward_rates.loc[forward_rates["Date"] < split_date]
    forward_rates_oos = forward_rates.loc[
        (forward_rates["Date"] >= split_date) & (forward_rates["Date"] <= end_date)
    ]

    if Macro_data is not None:
        Macro_data_insample = Macro_data.loc[Macro_data["Date"] < split_date]
        Macro_data_oos = Macro_data.loc[
            (Macro_data["Date"] >= split_date) & (Macro_data["Date"] <= end_date)
        ]
    else:
        Macro_data_insample, Macro_data_oos = None, None

    return {
        "excess_returns_insample": excess_returns_insample,
        "excess_returns_oos": excess_returns_oos,
        "forward_rates_insample": forward_rates_insample,
        "forward_rates_oos": forward_rates_oos,
        "Macro_data_insample": Macro_data_insample,
        "Macro_data_oos": Macro_data_oos,
    }

# Let's assume that this code is correct (for now)
def iterative_pca_regression(excess_returns_insample, forward_rates_insample,
                             excess_returns_oos, forward_rates_oos,
                             Macro_data_insample=None, Macro_data_oos=None):
    """
    Iteratively fits a regression on the first 3 principal components of the in-sample data,
    predicts the next out-of-sample value, then updates the model with that new data point.
    """

    pred_values = []

    # Scale and apply PCA on Macro_data if provided
    macro_pca_fit, macro_scaler = None, None
    if Macro_data_insample is not None:
        macro_scaler = StandardScaler().fit(Macro_data_insample)
        macro_insample_scaled = macro_scaler.transform(Macro_data_insample)
        macro_pca_fit = PCA(n_components=8).fit(macro_insample_scaled)
        macro_pcs_insample = macro_pca_fit.transform(macro_insample_scaled)
    else:
        macro_pcs_insample = None

    # PCA on forward_rates_insample
    n_components = 3
    pca = PCA(n_components).fit(forward_rates_insample)
    pcs_insample = pca.transform(forward_rates_insample)

    # Combine forward-rate PCs with Macro_data PCs
    if macro_pcs_insample is not None:
        X_insample = np.hstack([pcs_insample, macro_pcs_insample])
    else:
        X_insample = pcs_insample

    y_insample = excess_returns_insample
    model = LinearRegression().fit(X_insample, y_insample)

    # Iteration loop
    for i in range(len(excess_returns_oos)):
        # Transform new out-of-sample forward rates with existing PCA loadings
        test_fwd_pcs = pca.transform(forward_rates_oos.iloc[[i]])
        if macro_pca_fit is not None and Macro_data_oos is not None:
            test_macro_scaled = macro_scaler.transform(Macro_data_oos.iloc[[i]])
            test_macro_pcs = macro_pca_fit.transform(test_macro_scaled)
            test_X = np.hstack([test_fwd_pcs, test_macro_pcs])
        else:
            test_X = test_fwd_pcs

        # Predict and store as a scalar
        pred = model.predict(test_X)[0][0]  # Extract the scalar value
        pred_values.append(pred)

        # Add new data point to in-sample
        y_new = excess_returns_oos.iloc[[i]]
        X_new_fwd = forward_rates_oos.iloc[[i]]

        excess_returns_insample = pd.concat([excess_returns_insample, y_new])
        forward_rates_insample = pd.concat([forward_rates_insample, X_new_fwd])
        if macro_pca_fit is not None and Macro_data_oos is not None:
            new_macro = Macro_data_oos.iloc[[i]]
            Macro_data_insample = pd.concat([Macro_data_insample, new_macro])

        # Recompute PCA and re-fit model with the updated dataset
        pca = PCA(n_components).fit(forward_rates_insample)
        pcs_insample = pca.transform(forward_rates_insample)

        # Recompute PCA on Macro_data if provided
        if macro_pca_fit is not None:
            macro_scaler = StandardScaler().fit(Macro_data_insample)
            macro_insample_scaled = macro_scaler.transform(Macro_data_insample)
            macro_pca_fit = PCA(n_components=8).fit(macro_insample_scaled)
            macro_pcs_insample = macro_pca_fit.transform(macro_insample_scaled)
            X_insample = np.hstack([pcs_insample, macro_pcs_insample])
        else:
            X_insample = pcs_insample

        y_insample = excess_returns_insample
        model.fit(X_insample, y_insample)

    # Convert predictions to a pandas Series
    return pd.Series(pred_values, index=excess_returns_oos.index)

if __name__ == "__main__":
    # Load the data on forward rates and excess returns
    forward_rates = pd.read_excel("data-folder/Fwd rates and xr/forward_rates.xlsx")
    xr = pd.read_excel("data-folder/Fwd rates and xr/xr.xlsx") 
    macro_data = pd.read_excel("data-folder/Cleaned data/Yields+Final/Imputted_MacroData.xlsx")  # Example path

    start_oos = "1990-01-01"
    end_oos = "2018-12-01" # Keep it consistent with Bianchi

    # Convert start_oos to datetime
    start_oos_date = pd.to_datetime(start_oos)
    end_oos_date = pd.to_datetime(end_oos)

    # Extract insample and oos data into a dictionary
    dic = split_data_by_date(xr, forward_rates, start_oos_date, end_oos_date, Macro_data=macro_data)

    for key in [
        "excess_returns_insample",
        "excess_returns_oos",
        "forward_rates_insample",
        "forward_rates_oos",
        "Macro_data_insample",
        "Macro_data_oos"
    ]:
        dic[key] = dic[key].drop(columns="Date")
    

    excess_returns_insample = dic["excess_returns_insample"]
    excess_returns_oos = dic["excess_returns_oos"]
    forward_rates_insample = dic["forward_rates_insample"]
    forward_rates_oos = dic["forward_rates_oos"]
    Macro_data_insample = dic["Macro_data_insample"]
    Macro_data_oos = dic["Macro_data_oos"]

    # List of column names to iterate over
    columns_to_predict = ["2 y", "3 y", "4 y", "5 y", "7 y", "10 y"]

    # Dictionary to store predictions for each column
    predictions = {}


    # Run iterative PCA regression for each column
    for column in columns_to_predict:
        print(f"Running iterative PCA regression for column: {column}")

        # Extract the specific column for in-sample and out-of-sample excess returns
        excess_returns_insample_col = excess_returns_insample[[column]]
        excess_returns_oos_col = excess_returns_oos[[column]]

        # Run the regression
        pred_values = iterative_pca_regression(
            excess_returns_insample_col,
            forward_rates_insample,
            excess_returns_oos_col,
            forward_rates_oos,
            Macro_data_insample=Macro_data_insample,
            Macro_data_oos=Macro_data_oos,
        )

        # Store the predictions
        predictions[column] = pred_values

    
    ## Compute the out-of-sample R2
    # Compute benchmark predictions
    benchmark_predictions = compute_benchmark_prediction(
        excess_returns_insample, excess_returns_oos)

    
    for key in predictions.keys():
        print(f"Out-of-sample R2 for {key}: {r2_oos(excess_returns_oos[key], predictions[key], benchmark_predictions[key])}")
