import pandas as pd
import matplotlib.pyplot as plt

# Read the Excel file
file_path1 = "LW_monthly.xlsx"
file_path2 = "FRED-MD monthly 2024-12.csv"
Yields = pd.read_excel(file_path1, skiprows=8)
MacroData = pd.read_csv(file_path2, skiprows=0)
MacroData = MacroData.drop(0)

# Convert the Date column to datetime format
Yields["Date"] = pd.to_datetime(Yields["Date"].astype(str), format="%Y%m")
MacroData["sasdate"] = pd.to_datetime(MacroData["sasdate"])

#print(Yields["Date"].head(2))
#print(MacroData["sasdate"].head(2))
# Align the datasets based on the date range
start_date = max(Yields["Date"].min(), MacroData["sasdate"].min())
end_date = min(Yields["Date"].max(), MacroData["sasdate"].max())



Yields_aligned = Yields[(Yields["Date"] >= start_date) & (Yields["Date"] <= end_date)]
MacroData_aligned = MacroData[(MacroData["sasdate"] >= start_date) & (MacroData["sasdate"] <= end_date)]

# Save the aligned datasets to separate Excel files
Yields_aligned.to_excel("Aligned_Yields.xlsx", index=False)
MacroData_aligned.to_excel("Aligned_MacroData.xlsx", index=False)

# Function that plots yields with maturity m against the Date column 
def plot_column(m, df):
    plt.figure()
    plt.plot(df.iloc[:, 0], df.iloc[:, m]) 
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[m])
    plt.title(f'{df.columns[m]} over {df.columns[0]}')
    plt.show()

# Example usage
# plot_column(240, Yields_aligned)


