import pandas as pd
import numpy as np

# MAYBE INCORRECT CODE!!!!
def get_forward_rates(yield_df, yield_date_col='Date'):
    """
    Calculate forward rates from zero-coupon yields.

    Parameters:
    - yield_df: DataFrame containing zero-coupon yields with columns ['1 y', '2 y', ..., '10 y'].
    - yield_date_col: Name of the column containing dates.

    Returns:
    - DataFrame containing forward rates with the same date column and forward rate columns.
    """
    # Extract the maturity years from column names (e.g., '1 y' -> 1, '2 y' -> 2, etc.)
    maturities = [int(col.split()[0]) for col in yield_df.columns if col != yield_date_col]

    # Initialize a DataFrame to store forward rates
    forward_rates_df = pd.DataFrame()
    forward_rates_df[yield_date_col] = yield_df[yield_date_col]

    # Calculate forward rates using the formula:
    # f_t(n) = n * y_t(n) - (n-1) * y_t(n-1)
    for i in range(1, len(maturities)):
        n = maturities[i]
        prev_n = maturities[i - 1]
        forward_rate_col = f"{prev_n}-{n} y"
        forward_rates_df[forward_rate_col] = (
            n * yield_df[f"{n} y"] - prev_n * yield_df[f"{prev_n} y"]
        )

    return forward_rates_df

# INCORRECT CODE!!!!
def get_excess_returns(yield_df, yield_date_col='Date', risk_free_col='1 y'):
    """
    Calculate excess returns for bonds.

    Parameters:
    - yield_df: DataFrame containing zero-coupon yields with columns ['1 y', '2 y', ..., '10 y'].
    - yield_date_col: Name of the column containing dates.
    - risk_free_col: Name of the column containing the one-year risk-free rate.

    Returns:
    - DataFrame containing excess returns with the same date column and excess return columns.
    """
    # Extract the maturity years from column names (e.g., '1 y' -> 1, '2 y' -> 2, etc.)
    maturities = [int(col.split()[0]) for col in yield_df.columns if col != yield_date_col]

    # Initialize a DataFrame to store excess returns
    excess_returns_df = pd.DataFrame()
    excess_returns_df[yield_date_col] = yield_df[yield_date_col]

    # Calculate excess returns using the formula:
    # r_t+1(n) = log(P_t+1(n-1)) - log(P_t(n))
    # r_x_t+1(n) = r_t+1(n) - y_t(1)
    for i in range(1, len(maturities)):
        n = maturities[i]
        prev_n = maturities[i - 1]
        excess_return_col = f"Excess Return {n} y"
        excess_returns_df[excess_return_col] = (
            (prev_n * yield_df[f"{prev_n} y"].shift(-1) - n * yield_df[f"{n} y"])
            - yield_df[risk_free_col]
        )

    return excess_returns_df

if __name__ == "__main__":
    # Read the Excel file
    yield_df = pd.read_excel("data-folder/Aligned_Yields_Extracted.xlsx", parse_dates=True)
    #print(yield_df.columns)

    
    # Compute forward rates
    forward_rates = get_forward_rates(yield_df)
    excess_returns = get_excess_returns(yield_df)
    # Extract columns 0, 1, 2, 3, 4, 5, 7 and 10   
    forward_rates = forward_rates.iloc[:, [0, 1, 2, 3, 4, 6, 9]]
    excess_returns = excess_returns.iloc[:, [0, 1, 2, 3, 4, 6, 9]]
    forward_rates.to_excel("data-folder/Forward_Rates.xlsx", index=False)
    excess_returns.to_excel("data-folder/Excess_Returns.xlsx", index=False)
    print("Forward rates have been saved as 'Forward_Rates.xlsx'.")
    print("Excess returns have been saved as 'Excess_Returns.xlsx'.")
