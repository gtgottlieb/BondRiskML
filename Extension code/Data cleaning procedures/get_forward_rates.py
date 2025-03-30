import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def get_forward_rates(yield_df):
    """
    Implemented forward rate calculation according to Liu and Wu
    """
    date = yield_df.iloc[:, 0]  # First column is the date
    yield_df = yield_df.iloc[:, 1:]  # Exclude the date column

    logPt_n = pd.DataFrame(index=yield_df.index, dtype=np.float64)
    logPt_n_minus_1 = pd.DataFrame(index=yield_df.index, dtype=np.float64)
    forward_rates = pd.DataFrame(index=yield_df.index, dtype=np.float64)

    for n in range(1, len(yield_df.columns)):
        col_name_n = yield_df.columns[n]
        col_name_n_minus_1 = yield_df.columns[n - 1]
        
        logPt_n_minus_1[col_name_n] = -(n - 1) * yield_df[col_name_n_minus_1]
        logPt_n[col_name_n] = -n * yield_df[col_name_n]

        if n == 1:
            forward_rates[col_name_n] = yield_df[col_name_n]
        else:
            forward_rates[col_name_n] = logPt_n_minus_1[col_name_n] - logPt_n[col_name_n]


    # Add the date column to forward_rates
    forward_rates.insert(0, "Date", date)

    # Calculate excess return
    xr = pd.DataFrame(index=yield_df.index, dtype=np.float64)
    for n in range(1, len(yield_df.columns)):
        col_name_n = yield_df.columns[n]
        col_name_n_minus_1 = yield_df.columns[n - 1]

        if n == 1:
            xr[col_name_n] = 0
        else:
            rt_plus_1 = logPt_n_minus_1[col_name_n].shift(-12) - logPt_n[col_name_n]
            xr[col_name_n] = rt_plus_1 - yield_df.iloc[:, 1]

    # Add the date column to xr
    xr.insert(0, "Date", date)

    return forward_rates, xr


if __name__ == "__main__":
    # Read the Excel file
    yield_df = pd.read_excel("data-folder/CP replication data/Yields.xlsx", parse_dates=True)
    date_col = yield_df.columns[0]

    forward_rates, xr = get_forward_rates(yield_df)
    


    forward_rates.columns = ["Date", "1 year", "2 years", "3 years", "4 years", "5 years"]
    xr.columns = ["Date", "1 year", "2 years", "3 years", "4 years", "5 years"]

    # Save to excel
    forward_rates.to_excel("data-folder/CP replication data/forward_rates.xlsx", index=False)
    xr.to_excel("data-folder/CP replication data/xr.xlsx", index=False)

    ## Getting the data ready for analysis
    start_date = "1964-01-01"
    end_date = "2003-01-01"

    # Filter the data for the specified date range
    forward_rates = forward_rates[(forward_rates['Date'] >= start_date) & (forward_rates['Date'] <= end_date)]
    xr = xr[(xr['Date'] >= start_date) & (xr['Date'] <= end_date)]
    # Extract only the columns 2-5 ["2 years", "3 years", "4 years", "5 years"] for xr
    # Extract only the columns 1-5 ["1 year", "2 years", "3 years", "4 years", "5 years"] for forward_rates
    xr = xr.iloc[:, 2:6]
    xr['Row_Average'] = xr.mean(axis=1) # Compute row average for each row in xr
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

    print("Regression coefficients (PCA):\n")
    print("Regression coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    print("R-squared:", model.score(X, y)) 

    # Regress directly on forward rates
    X = forward_rates

    model2 = LinearRegression()
    model2.fit(X, y)

    # Print the regression coefficients  
    print("\nRegression coefficients (Forward Rates):\n") 
    print("Regression coefficients:", model2.coef_)
    print("Intercept:", model2.intercept_)
    print("R-squared:", model2.score(X, y))

