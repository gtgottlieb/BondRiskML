import pandas as pd

macro_data_final = pd.read_excel("data-folder/Cleaned data/Yields+Final/Aligned_MacroData.xlsx")
macro_data_final.rename(columns={'sasdate': 'Date'}, inplace=True)
# Set 'Date' as the index for time-based interpolation
macro_data_final.set_index('Date', inplace=True)

# Summarize missing values for columns with NaNs
missing_values = macro_data_final.isna().sum()
missing_columns = missing_values[missing_values > 0]
print("Columns with missing values:")
print(missing_columns)

# Calculate percentage of missing values for those columns
missing_percentage = macro_data_final.isna().mean() * 100
missing_percentage = missing_percentage[missing_percentage > 0]
print("\nPercentage of missing values for columns with NaNs:")
print(missing_percentage)

# Display rows with any NaNs
nan_rows = macro_data_final[macro_data_final.isna().any(axis=1)]
print("\nRows with missing values:")
print(nan_rows)



# Forward-fill and backward-fill for columns with very few missing values
columns_to_ffill = ['CMRMTSPLx', 'HWI', 'HWIURATIO', 'BUSINVx', 'ISRATIOx', 
                    'NONREVSL', 'CONSPI', 'S&P PE ratio', 'CP3Mx', 'COMPAPFFx', 
                    'DTCOLNVHFNM', 'DTCTHFNM']
macro_data_final[columns_to_ffill] = macro_data_final[columns_to_ffill].ffill().bfill()

# Time-based interpolation for columns with moderate missing values
columns_to_interpolate = ['TWEXAFEGSMTHx', 'UMCSENTx']
macro_data_final[columns_to_interpolate] = macro_data_final[columns_to_interpolate].interpolate(method='time')

# Drop ACOGNO
macro_data_final = macro_data_final.drop(columns=['ACOGNO'])

# Forward-fill and backward-fill for S&P Dividend Yield
macro_data_final['S&P div yield'] = macro_data_final['S&P div yield'].ffill().bfill()

macro_data_final.reset_index(inplace=True)

# Save the cleaned macro data to an Excel file
macro_data_final.to_excel("data-folder/Cleaned data/Yields+Final/Imputted_MacroData.xlsx", index=False)
