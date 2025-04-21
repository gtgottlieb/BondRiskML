import pandas as pd
import matplotlib.pyplot as plt

xr = pd.read_excel("data-folder/!Data for forecasting/xr.xlsx")
forward_rates = pd.read_excel("data-folder/!Data for forecasting/forward_rates.xlsx")

date_xr = pd.to_datetime(xr['Date'])
date_forward_rates = pd.to_datetime(forward_rates['Date'])

plt.figure(figsize=(10, 6))

# Plotting excess returns for zero-coupon Treasury bonds
for column in xr.iloc[:, [2,5]].columns:
    plt.plot(date_xr, xr[column], label=column)

plt.title("Excess Returns of zero-coupon Treasury bonds")
plt.xlabel("Date")
plt.ylabel("Excess Return")
plt.legend()
plt.grid(True)
plt.show()

#Plotting forward rates
plt.figure(figsize=(10, 6))
for column in forward_rates.iloc[:, [2,5,10]].columns:
    plt.plot(date_forward_rates, forward_rates[column], label=column)

plt.title("One year forward rates of zero-coupon Treasury bonds")
plt.xlabel("Date")
plt.ylabel("Forward Rate")
plt.legend()
plt.grid(True)
plt.show()

# Plotting forecasts

