import pandas as pd

forward_rates = pd.read_excel("data-folder/Forward_rates.xlsx", parse_dates=True)
xr = pd.read_excel("data-folder/Excess_return.xlsx", parse_dates=True)

# Filter the data for the specified date range
start_date = "1964-01-01"
end_date = "2003-01-01"

## Getting the data ready for analysis

# Filter the data for the specified date range
forward_rates = forward_rates[(forward_rates['Date'] >= start_date) & (forward_rates['Date'] <= end_date)]
xr = xr[(xr['Date'] >= start_date) & (xr['Date'] <= end_date)]
# Extract only the columns 2-5 ["2 years", "3 years", "4 years", "5 years"] for xr
# Extract only the columns 1-5 ["1 year", "2 years", "3 years", "4 years", "5 years"] for forward_rates
xr = xr.iloc[:, 2:5]
forward_rates = forward_rates.iloc[:, 1:5]

print(xr.columns)
print(forward_rates.columns)


