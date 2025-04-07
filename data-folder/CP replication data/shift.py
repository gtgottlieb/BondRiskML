import pandas as pd

# Load the Excel file into a DataFrame
xr = pd.read_excel("data-folder/CP replication data/xr.xlsx")

# Shift only the 'Date' column by -12
xr['Date'] = xr['Date'].shift(-12)
xr.to_excel("data-folder/CP replication data/xr_test.xlsx", index=False)