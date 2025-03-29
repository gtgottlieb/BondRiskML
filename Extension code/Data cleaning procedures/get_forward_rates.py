import pandas as pd
import numpy as np


def get_forward_rates(yield_df):
    """
    Implemented forward rate calculation according to Liu and Wu
    """
    # Equation 5.1 -> correct
    logPt_n = pd.DataFrame(index=yield_df.index, dtype=np.float64)
    logPt_n_minus_1 = pd.DataFrame(index=yield_df.index, dtype=np.float64)
    # We start the loop from 2 because the first column of forward rates is the first yield
    for n in range(2, len(yield_df.columns)):
        logPt_n_minus_1[n] = -(n-1) * yield_df.iloc[:, n-1]
        logPt_n[n] = -n * yield_df.iloc[:, n] 
        
    # Calculate the forward rates
    # Equation 5.2 -> correct
    forward_rates = pd.DataFrame(index=yield_df.index, dtype=np.float64)
    for n in range(0, len(yield_df.columns) - 1):
        if n == 0:
            forward_rates[n] = yield_df.iloc[:, n] # Add the date column
        elif n == 1:
            forward_rates[n] = yield_df.iloc[:, n] # Add the 1 year yield column
        else:
            forward_rates[n] = logPt_n_minus_1[n] - logPt_n[n] # Calculate forward rate according to 5.2
   
    
    # Equation 5.3 and 5.4
    # Calculating the excess return
    
    xr = pd.DataFrame(index=yield_df.index, dtype=np.float64)
    for n in range(0, len(yield_df.columns) - 1):
        if n == 0:
            xr[n] = yield_df.iloc[:, n] # Add the date column
        elif n == 1:
            xr[n] = 0  # Add a column of 0s
        else:
            for t in range(0, len(yield_df) - 1):
                rt_plus_1 = logPt_n_minus_1.loc[t+1, n] - logPt_n.loc[t, n]  # Calculate one year holding period return
                xr.loc[t, n] = rt_plus_1 - yield_df.iloc[t, 1]  # Calculate the excess return
    

    return forward_rates, xr

if __name__ == "__main__":
    # Read the Excel file
    yield_df = pd.read_excel("data-folder/Aligned_Yields_Extracted.xlsx", sheet_name="Forward_rates", parse_dates=True)
    #print(yield_df.columns)
    forward_rates, xr = get_forward_rates(yield_df)
    # Add column names to the forward rates DataFrame
    
    '''
    xr.columns = ["Date", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years",
    "9 years", "10 years"]
    # Save the excess return to a new Excel file
    xr.to_excel("data-folder/Excess_return.xlsx", index=False)

    forward_rates.columns = ["Date", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years",
    "9 years", "10 years"]
    # Save the forward rates to a new Excel file
    forward_rates.to_excel("data-folder/Forward_rates.xlsx", index=False)
    '''