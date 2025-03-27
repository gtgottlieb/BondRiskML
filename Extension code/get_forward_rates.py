import pandas as pd
import numpy as np

def get_forward_rates(yield_df, yield_date_col='Date'):
    """
    Compute forward rates from zero-coupon yields.

    Parameters:
    - yield_df: DataFrame containing zero-coupon yields. The first column should be the date,
      and the other columns should represent maturities in months.
    - yield_date_col: Name of the column containing the dates.

    Returns:
    - DataFrame with forward rates. The structure is similar to the input DataFrame.
    """
    # Ensure the date column is in datetime format
    yield_df[yield_date_col] = pd.to_datetime(yield_df[yield_date_col])

    # Extract maturity columns (skip the date column)
    maturity_columns = yield_df.columns[1:]  # Skip the first column (Date)

    # Initialize a DataFrame to store forward rates
    forward_rates = pd.DataFrame({yield_date_col: yield_df[yield_date_col]})

    # Compute forward rates
    for i in range(1, len(maturity_columns)):
        # Use maturity columns directly
        P_n = np.exp(-int(maturity_columns[i].split()[0]) / 12 * yield_df[maturity_columns[i]])
        P_n_minus_1 = np.exp(-int(maturity_columns[i - 1].split()[0]) / 12 * yield_df[maturity_columns[i - 1]])

        # Equation 5.2: Compute forward rates
        f_n = (np.log(P_n_minus_1) - np.log(P_n)) * 12  # Annualized forward rate
        forward_rates[maturity_columns[i]] = f_n

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
    forward_rates.to_excel("data-folder/Cleaned data/Yields+Final/Forward_Rates.xlsx", index=False)
    print("Forward rates have been saved as 'Forward_Rates.xlsx'.")

    # Extract necessary columns
    extract_data("data-folder/Cleaned data/Yields+Final/Forward_Rates.xlsx", "data-folder/Extracted_fwd_rates.xlsx")
