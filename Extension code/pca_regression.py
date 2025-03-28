from sklearn.decomposition import PCA
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from Roos import compute_oos_r2

def split_data_by_date(excess_returns, forward_rates, split_date, end_date):
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

    return {
        "excess_returns_insample": excess_returns_insample,
        "excess_returns_oos": excess_returns_oos,
        "forward_rates_insample": forward_rates_insample,
        "forward_rates_oos": forward_rates_oos,
    }

def iterative_pca_regression(excess_returns_insample, forward_rates_insample,
                             excess_returns_oos, forward_rates_oos):
    """
    Iteratively fits a regression on the first 3 principal components of the in-sample data,
    predicts the next out-of-sample value, then updates the model with that new data point.
    """

    i = 0
    pred_values = []

    # Initial PCA on forward_rates_insample
    pca = PCA(n_components=3)
    pca_fit = pca.fit(forward_rates_insample)
    pcs_insample = pca_fit.transform(forward_rates_insample)
    y_insample = excess_returns_insample

    # Initial regression using LinearRegression
    model = LinearRegression()
    model.fit(pcs_insample, y_insample)

    # Iteration loop

    for i in range(len(excess_returns_oos)):
        # Transform new out-of-sample forward rates with existing PCA loadings
        test_pcs = pca_fit.transform(forward_rates_oos.iloc[[i]])

        # Predict and store as a scalar
        pred = model.predict(test_pcs)[0][0]  # Extract the scalar value
        pred_values.append(pred)

        # Add new data point to in-sample
        y_new = excess_returns_oos.iloc[[i]]
        X_new_fwd = forward_rates_oos.iloc[[i]]

        excess_returns_insample = pd.concat([excess_returns_insample, y_new])
        forward_rates_insample = pd.concat([forward_rates_insample, X_new_fwd])

        # Recompute PCA and re-fit model with the updated dataset
        pca_fit = PCA(n_components=3).fit(forward_rates_insample)  # Ensure consistent number of components
        pcs_insample = pca_fit.transform(forward_rates_insample)
        y_insample = excess_returns_insample

        model = LinearRegression()
        model.fit(pcs_insample, y_insample)

    # Convert predictions to a pandas Series
    return pd.Series(pred_values, index=excess_returns_oos.index)

if __name__ == "__main__":
    # Load data without the first 13 rows
    forward_rates = pd.read_excel(
        "data-folder/Extracted_fwd_rates_new.xlsx", parse_dates=True, header=0
    ).iloc[13:, :].reset_index(drop=True)
    excess_returns = pd.read_excel(
        "data-folder/Extracted_excess_returns.xlsx", parse_dates=True, header=0
    ).iloc[13:, :].reset_index(drop=True)
    start_oos = "1990-01-01"
    end_oos = "2018-12-01" # Keep it consistent with Bianchi

    # Convert start_oos to datetime
    start_oos_date = pd.to_datetime(start_oos)
    end_oos_date = pd.to_datetime(end_oos)

    # Extract insample and oos data into a dictionary
    dic = split_data_by_date(excess_returns, forward_rates, start_oos_date, end_oos_date)

    for key in [
        "excess_returns_insample",
        "excess_returns_oos",
        "forward_rates_insample",
        "forward_rates_oos",
    ]:
        dic[key] = dic[key].drop(columns="Date")

    excess_returns_insample = dic["excess_returns_insample"]
    excess_returns_oos = dic["excess_returns_oos"]
    forward_rates_insample = dic["forward_rates_insample"]
    forward_rates_oos = dic["forward_rates_oos"]

    # List of column names to iterate over
    columns_to_predict = ["24 m", "48 m", "60 m", "84 m", "120 m"]

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
        )

        # Store the predictions
        predictions[column] = pred_values

    
    # Compute the out-of-sample R2
    for key in predictions.keys():
        print(f"Out-of-sample R2 for {key}: {compute_oos_r2(excess_returns_oos[key], predictions[key])}")
    