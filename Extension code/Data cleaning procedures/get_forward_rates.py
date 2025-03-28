import pandas as pd
import numpy as np


def get_forward_rates(yield_df):
    """
    Implemented forward rate calculation according to Liu and Wu
    """
    # Equation 5.1 -> correct
    logPt_n = pd.DataFrame(index=yield_df.index, dtype=np.float64)
    logPt_n_minus_1 = pd.DataFrame(index=yield_df.index, dtype=np.float64)
    # We start the loop from 2 because the first column in forward rates is the first yield
    for col in range(2, len(yield_df.columns)):
        logPt_n_minus_1[col] = -(col-1) * yield_df.iloc[:, col-1]
        logPt_n[col] = -col * yield_df.iloc[:, col] 
        
    # Calculate the forward rates
    # Equation 5.2 -> correct
    forward_rates = pd.DataFrame(index=yield_df.index, dtype=np.float64)
    for col in range(0, 11):
        if col == 0:
            forward_rates[col] = yield_df.iloc[:, col] 
        elif col == 1:
            forward_rates[col] = yield_df.iloc[:, col]
        else:
            forward_rates[col] = logPt_n_minus_1[col] - logPt_n[col]
   
    return forward_rates

if __name__ == "__main__":
    # Read the Excel file
    yield_df = pd.read_excel("data-folder/Aligned_Yields_Extracted.xlsx", sheet_name="Forward_rates", parse_dates=True)
    #print(yield_df.columns)
    forward_rates = get_forward_rates(yield_df)
    # Extract specific columns from the forward rates
    forward_rates = forward_rates.iloc[:, [0, 1, 2, 3, 4, 5, 7, 10]]
    # Add column names to the forward rates DataFrame
    forward_rates.columns = ["Date", "1 year", "2 years", "3 years", "4 years", "5 years", "7 years", "10 years"]
    # Save the forward rates to a new Excel file
    forward_rates.to_excel("data-folder/Forward_rates.xlsx", index=False)