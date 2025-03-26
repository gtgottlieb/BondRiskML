import pandas as pd
import numpy as np
import os


"""
This script uses the original LW_monthly data to compute excess returns for 12-month holding periods.
Not: Since data from before 1971-08-01 is not available for the 120m maturity, 
excess returns for the 120m maturity are only available from 1972-08-01 onwards.
"""
def compute_excess_returns_lw(filepath, output_path, max_maturity_months=120):
    """
    Compute CP-style 12-month excess returns from Liu & Wu zero-coupon yields.

    Parameters:
    - filepath: path to the Excel file with LW yield data
    - output_path: where to save the resulting Excel file
    - max_maturity_months: maximum bond maturity in months to include
    """

    # Load the data, skipping metadata rows
    df = pd.read_excel(filepath, skiprows=5)
    df.dropna(axis=0, how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    # Rename date column
    df.rename(columns={df.columns[0]: 'date'}, inplace=True)

    # Remove any rows where the 'date' column isn't a valid date
    df = df[df['date'].apply(lambda x: str(x).isdigit())].copy()

    # Convert 'date' from YYYYMM to datetime
    df['date'] = pd.to_datetime(df['date'].astype(int).astype(str), format='%Y%m')

    # Select only columns for maturities 1m to 120m
    maturity_columns = df.columns[1:max_maturity_months+1]  # '1m' is second column
    yields = df[maturity_columns].values / 100.0  # Convert percentage to decimal
    maturities = np.arange(1, max_maturity_months + 1)  # in months

    # Compute log bond prices: p_t^{(n)} = - (n/12) * y_t^{(n)}
    prices = -yields * (maturities / 12.0)

    # Initialize excess return array
    T = yields.shape[0]
    excess_returns = np.full((T, max_maturity_months), np.nan)

    # Compute: xr_t^{(n)} = p_{t+12}^{(n-12)} - p_t^{(n)} - y_t^{(12)}
    for n in range(12, max_maturity_months + 1):
        p_future = prices[12:, n - 12]     # p_{t+12}^{(n-12)}
        p_now = prices[:-12, n - 1]        # p_t^{(n)}
        y_12 = yields[:-12, 11]            # y_t^{(12)}
        excess_returns[12:, n - 1] = p_future - p_now - y_12

    # Create output DataFrame
    xr_columns = [f"{n}m" for n in range(1, max_maturity_months + 1)]
    df_excess = pd.DataFrame(excess_returns, columns=xr_columns)
    df_excess['date'] = df['date'].values
    df_excess = df_excess[['date'] + xr_columns]  # reorder columns

    # Save to Excel
    df_excess.to_excel(output_path, index=False)
    print(f"Excess returns saved to: {output_path}")

def extract_data(filepath, output_path):
    """
    Extract data from the Excel file with excess returns.

    Parameters:
    - filepath: path to the Excel file with excess returns
    - output_path: where to save the resulting Excel file
    """

    # Load the data, ignoring the rows before the date 1971-08-01
    df = pd.read_excel(filepath)
    df = df[df['date'] >= '1971-08-01']

    # Extract the columns with excess returns
    indices = ["24m", "36m", "48m", "60m", "84m", "120m"]

    # Create a new DataFrame with the extracted columns
    df_extracted = df[['date'] + indices].copy()

    # Save to Excel
    df_extracted.to_excel(output_path, index=False)
    print(f"Extracted data saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    input_path = os.path.join("data-folder", "Raw data", "LW_monthly.xlsx")
    output_path = os.path.join("data-folder", "Cleaned data", "Yields+Final", "Excess_Returns.xlsx")

    extracted_path = os.path.join("data-folder", "Extracted_excess_returns.xlsx")

    compute_excess_returns_lw(input_path, output_path)
    extract_data(output_path, extracted_path)
