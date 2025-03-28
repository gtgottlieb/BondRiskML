import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def get_forward_rates(yield_df, yield_date_col='Date'):
    """
    Compute 1-year forward rates from zero-coupon yields, and label each
    forward by its ending maturity in months (e.g. '13 m', '24 m', etc.).

    The DataFrame `yield_df` should have:
      - A 'Date' column,
      - Columns for each maturity in months (e.g. '1 m', '2 m', ..., '120 m'),
        where each cell is the annualized, continuously-compounded yield
        in PERCENTAGE POINTS (e.g. 5.0 = 5%).
    Steps:
      1) Convert yields (in % points) to decimals (5.0 -> 0.05).
      2) Compute log-price p(t,m) = -(m/12)*y(t,m),
         because m/12 is the maturity in years, y(t,m) is the decimal yield.
      3) 1-year forward from m to m+12 is p(t,m) - p(t,m+12).
      4) Label the new column as '{m+12} m'.
    """

    df = yield_df.copy()
    df[yield_date_col] = pd.to_datetime(df[yield_date_col])

    # Identify maturity columns
    maturity_cols = df.columns.drop(yield_date_col)

    # Build a map from integer months -> original column name
    months_map = {}
    for col in maturity_cols:
        # e.g. '24 m' -> split -> ['24','m'] -> m_int=24
        m_int = int(col.split()[0])
        months_map[m_int] = col

    sorted_months = sorted(months_map.keys())

    # Initialize output with the Date column
    forward_rates = pd.DataFrame({yield_date_col: df[yield_date_col]})

    # For each maturity m, see if m+12 exists in data. If so, compute forward
    for m in sorted_months:
        m_next = m + 12
        if m_next in months_map:
            # Convert yields from % points to decimal
            y_m     = df[months_map[m]]     / 100.0
            y_mnext = df[months_map[m_next]] / 100.0

            # Log-price
            p_m     = -(m/12.0)      * y_m
            p_mnext = -((m_next)/12.0)* y_mnext

            # 1-year forward from m to m+12
            f_m = p_m - p_mnext

            # Label the column by the final maturity, e.g. '24 m'
            new_col = f"{m_next} m"
            forward_rates[new_col] = f_m

    return forward_rates


def extract_data(filepath, output_path):
    """
    Extract data from the Excel file with excess returns.

    Parameters:
    - filepath: path to the Excel file with excess returns
    - output_path: where to save the resulting Excel file
    """

    # Load the data, ignoring the rows before the date 1971-08-01
    df = pd.read_excel(filepath)

    df.columns = [c.strip() for c in df.columns]

    # Extract the columns with excess returns
    indices = ["24 m", "36 m", "48 m", "60 m", "84 m", "120 m"]

    # Create a new DataFrame with the extracted columns
    df_extracted = df[['Date'] + indices].copy()

    # Save to Excel
    df_extracted.to_excel(output_path, index=False)
    print(f"Extracted data saved to: {output_path}")

if __name__ == "__main__":
    # Read the Excel file
    yield_df = pd.read_excel("data-folder/Cleaned data/Yields+Final/Aligned_Yields.xlsx", parse_dates=True)

    # Compute forward rates
    forward_rates = get_forward_rates(yield_df)
    forward_rates.to_excel("data-folder/Cleaned data/Yields+Final/Forward_Rates_new.xlsx", index=False)
    print("Forward rates have been saved as 'Forward_Rates_new.xlsx'.")

    # Extract necessary columns
    extract_data("data-folder/Cleaned data/Yields+Final/Forward_Rates_new.xlsx", "data-folder/Extracted_fwd_rates_new.xlsx")
