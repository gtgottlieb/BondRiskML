import pandas as pd

yields_df = pd.read_excel("data-folder/Raw data/LW_monthly.xlsx", parse_dates=True, skiprows=8)
yields_df.columns = yields_df.columns.str.strip()
cols_to_extract = ["Date", "12 m", "24 m", "36 m", "48 m", "60 m", "72 m"]
yields_df = yields_df[cols_to_extract]
yields_df.columns = ["Date", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years"]
# Convert Excel serial dates to proper datetime format
yields_df["Date"] = pd.to_datetime(yields_df["Date"].astype(str), format="%Y%m")

print(yields_df.head())
# Save to new Excel file
yields_df.to_excel("data-folder/CP replication data/Yields.xlsx", index=False)

