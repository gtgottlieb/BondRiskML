import pandas as pd
import numpy as np
import os

def calculate_excess_returns(df_yields):
    """
    Compute log excess returns from a DataFrame of annualized yields.
    
    Parameters
    ----------
    df_yields : pd.DataFrame
        DataFrame of shape (T x M) with columns for each bond maturity in months.
        Example columns: '1 m', '2 m', ..., '120 m'.
        Each entry is the annualized yield (decimal, e.g. 0.05 for 5%) at time t.
        The DataFrame index is time (monthly or otherwise).
    
    Returns
    -------
    df_excess : pd.DataFrame
        DataFrame of the same shape as df_yields (minus edge effects),
        containing log excess returns. Rows aligned with df_yields.index,
        columns aligned with the same maturities, but note that for the very
        last row or last maturity, returns will be NaN where calculation is not
        feasible (e.g., n=1 cannot form n-1=0-month bond).
    """
    
    # 1) Convert yields to log prices p_t^(n) = -(n/12)* y_t^(n)
    #    We'll parse out the integer from the column name (e.g. "15 m" -> 15).
    maturities = []
    for col in df_yields.columns:
        # Extract the number of months from column label
        # e.g. '10 m' -> 10
        months_str = col.split()[0]  # everything before the space
        months = int(months_str)
        maturities.append(months)
    # Sort the columns by their numeric maturity so we can shift columns easily
    sorted_cols = [c for _, c in sorted(zip(maturities, df_yields.columns), key=lambda x: int(x[0]))]
    df_sorted = df_yields[sorted_cols].copy()
    numeric_maturities = sorted(maturities)

    # Convert yields to log prices
    # p^(n)_t = - (n/12) * y^(n)_t
    df_prices = df_sorted.mul(-1)
    for i, col in enumerate(df_sorted.columns):
        n = numeric_maturities[i]  # months
        df_prices[col] = df_prices[col] * (n / 12.0)

    # 2) For each n, the log return from t to t+1 is p_{t+1}^{(n-1)} - p_t^{(n)}.
    #    We'll shift the "price" DataFrame by -1 row (time) AND also shift columns
    #    so that for each n we find p_{t+1}(n-1).
    df_prices_tplus1 = df_prices.shift(-1)  # shift time up by 1
    # We'll also shift columns to align p_{t+1}^{(n-1)} in the same column as p_t^{(n)}:
    # method: create a 2D array with the same shape, fill with NaNs, then place the shifted columns in
    arr_prices_shifted = np.full_like(df_prices_tplus1.values, np.nan, dtype=float)

    # For column i (maturity n), we want to read column (i-1) from df_prices_tplus1, because that's n-1.
    for i in range(len(numeric_maturities)):
        n = numeric_maturities[i]
        if n == 1:
            # n=1 means no n-1=0 column, can't compute a next price for a 0-month bond
            continue
        else:
            arr_prices_shifted[:, i] = df_prices_tplus1.iloc[:, i-1]

    df_prices_nminus1_tplus1 = pd.DataFrame(arr_prices_shifted,
                                            index=df_prices.index,
                                            columns=df_prices.columns)

    # log returns: r_t+1^(n) = p_{t+1}(n-1) - p_t(n)
    df_log_returns = df_prices_nminus1_tplus1.sub(df_prices)

    # 3) Subtract the short rate to get excess returns:
    # short rate at time t is y_t^(1) / 12 in log approximation
    # We'll retrieve that from df_sorted['1 m'], if it exists
    if '1 m' not in df_sorted.columns:
        raise KeyError("We assume the short rate is the '1 m' column, but not found.")
    # annualized short rate => monthly log is y_(1)/12
    # We'll broadcast-subtract from each column
    df_excess = df_log_returns.sub(df_sorted['1 m'] / 12.0, axis=0)

    # For n=1, no valid return because it becomes 0-month next period, so set them to NaN
    df_excess['1 m'] = np.nan

    # The final row is also NaN because shifting -1 goes out of range
    # That is already handled from the shift above.

    return df_excess

if __name__ == "__main__":
    input_path = os.path.join("data-folder", "Cleaned data", "Yields+Final", "Aligned_Yields.xlsx")
    output_path = os.path.join("data-folder", "Cleaned data", "Yields+Final", "Excess_Returns.xlsx")

    df_in = pd.read_excel(input_path, index_col=0)

    # clean up column names:
    df_in.columns = [c.strip() for c in df_in.columns]

    df_excess = calculate_excess_returns(df_in)
    df_excess.to_excel(output_path)
