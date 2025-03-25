import pandas as pd
excess_returns = pd.read_excel("data-folder/Cleaned data/Yields+Final/Excess_Returns.xlsx", index_col=0, parse_dates=True)
excess_returns = excess_returns.loc["1984-01-01":"2018-06-01"]
columns_to_extract = [f"{i} m" for i in range(12, 121, 12)]
excess_returns = excess_returns[columns_to_extract]

annualized_mean = excess_returns.mean() * 12
annualized_std = excess_returns.std() * (12 ** 0.5)

print("Annualized Mean:")
print(annualized_mean)
print("\nAnnualized Std:")
print(annualized_std)