import pandas as pd

macro_data = pd.read_excel("data-folder/Cleaned data/Yields+Final/Aligned_MacroData.xlsx")
macro_data = macro_data.dropna(axis=1)
macro_data.to_excel("data-folder/!Data for forecasting/Imputted_Macro2.xlsx", index=False)