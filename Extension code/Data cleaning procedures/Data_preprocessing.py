import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression



def process_yield_data(yields_df):
    """
    Process the yield data to calculate logPt_n, forward rates, and excess returns (xr),
    keeping the date column intact.

   Args:
    yields_df (pd.DataFrame): DataFrame containing yield data with the first column as dates.

Returns:
    tuple: A tuple containing two DataFrames - forward_rates and xr
"""
    date = yields_df.iloc[:, 0]
    yields_df = yields_df.iloc[:, 1:]

    logPt_n = pd.DataFrame(index=yields_df.index, dtype=np.float64)
    forward_rates = pd.DataFrame(index=yields_df.index, dtype=np.float64)
    xr = pd.DataFrame(index=yields_df.index, dtype=np.float64)

    for n, col in enumerate(yields_df.columns):
        # Calculate logPt_n
        logPt_n[col] = -(n + 1) * yields_df[col]
        # Calculate forward rates
        if n == 0:
            forward_rates[col] = yields_df[col]
            xr[col] = 0
        else:
            # Calculate the forward rate 
            forward_rates[col] = logPt_n.iloc[:, n - 1] - logPt_n.iloc[:, n]
            # Calculate excess return
            rt_plus_1 = logPt_n.iloc[:, n - 1].shift(-12) - logPt_n.iloc[:, n]
            xr[col] = rt_plus_1 - yields_df.iloc[:, 0]

    forward_rates.insert(0, "Date", date)
    xr.insert(0, "Date", date.shift(-12))

    # Save to excel files
    #forward_rates.to_excel("data-folder/forward_rates.xlsx", index=False)
    #xr.to_excel("data-folder/xr_test.xlsx", index=False)

    return forward_rates, xr

def run_cp_replication(yields_df, start_date="1964-01-01", end_date="2019-01-01"):
    """
    Run the CP replication process on the given yield data DataFrame.

    Args:
        yields_df (pd.DataFrame): DataFrame containing yield data.
        start_date (str): Start date for filtering the data (default: "1964-01-01").
        end_date (str): End date for filtering the data (default: "2019-01-01").
    """
    
    xr_start_date = (pd.to_datetime(start_date) + pd.DateOffset(years=1)).strftime('%Y-%m-%d')
    xr_end_date = (pd.to_datetime(end_date) + pd.DateOffset(years=1)).strftime('%Y-%m-%d')

    # Process the yield data
    forward_rates, xr = process_yield_data(yields_df)

    # Filter the data for the specified date range
    forward_rates = forward_rates[(forward_rates['Date'] >= start_date) & (forward_rates['Date'] <= end_date)]
    xr = xr[(xr['Date'] >= xr_start_date) & (xr['Date'] <= xr_end_date)]

    xr = xr.iloc[:, 2:6]
    xr['Row_Average'] = xr.mean(axis=1)  # Compute row average for each row in xr
    forward_rates = forward_rates.iloc[:, 1:6]

    # Perform PCA on forward_rates
    
    pca = PCA(n_components=5)    
    principal_components = pca.fit_transform(forward_rates)

    # Create a DataFrame for the principal components
    principal_df = pd.DataFrame(principal_components, index=forward_rates.index, columns=[f"PC{i+1}" for i in range(5)])

    # Regress xr['Row_Average'] on the first 5 principal components
    X = principal_df
    y = xr['Row_Average']


    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    print("Regression coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    print("R-squared:", model.score(X, y))

if __name__ == "__main__":
    # Example usage with a DataFrame
    file_path = "data-folder/CP replication data/Yields.xlsx"
    yields_df = pd.read_excel(file_path, parse_dates=True)
    run_cp_replication(yields_df)



