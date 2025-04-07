import pandas as pd

macro_data = pd.read_excel("data-folder/!Data for forecasting/Imputted_MacroData2.xlsx")
# Winsorize numeric columns by clipping extreme values at the 5th and 95th percentiles.
for col in macro_data.select_dtypes(include="number").columns:
    lower = macro_data[col].quantile(0.05)
    upper = macro_data[col].quantile(0.95)
    macro_data[col] = macro_data[col].clip(lower=lower, upper=upper)

# Save the winsorized data back to the same Excel file.
macro_data.to_excel("data-folder/!Data for forecasting/Imputted_MacroData2.xlsx", index=False)

