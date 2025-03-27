import pandas as pd
import numpy as np
#from NNFuncBib import ElasticNet_Exog_Plain
from NNFuncBib import ElasticNet_NoMacro_Plain
from sklearn.metrics import mean_squared_error
import os

# === SETTINGS ===
EXCESS_RETURN_PATH = "data-folder/Extracted_excess_returns.xlsx"
FWD_RATE_PATH = "data-folder/Extracted_fwd_rates_new.xlsx"
OOS_START_DATE = "1990-01-01"
OOS_END_DATE = "2018-12-01"
FULL_START_DATE = "1971-08-01"  # for full sample filtering

# === Load and align data ===
df_xr = pd.read_excel(EXCESS_RETURN_PATH)
df_fwd = pd.read_excel(FWD_RATE_PATH)

# Normalize column names
df_xr.columns = [c.strip().lower() for c in df_xr.columns]
df_fwd.columns = [c.strip().lower() for c in df_fwd.columns]

df_xr['date'] = pd.to_datetime(df_xr['date'])
df_fwd['date'] = pd.to_datetime(df_fwd['date'])

df = pd.merge(df_xr, df_fwd, on="date", suffixes=("_xr", "_fwd"))
df.set_index("date", inplace=True)

# Define maturities and columns
maturities = ["24 m", "36 m", "48 m", "60 m", "84 m", "120 m"]
target_cols = [m + "_xr" for m in maturities]
fwd_cols = [m + "_fwd" for m in maturities]

# Apply forecast horizon shift (vintage consistency)
for col in target_cols:
    df[col] = df[col].shift(-12)
# Drop rows with any missing values
df.dropna(inplace=True)

# Limit full data to training + OOS period
df = df[(df.index >= FULL_START_DATE) & (df.index <= OOS_END_DATE)].copy()

# === Prepare arrays ===
Y = df[target_cols].values
Xexog = df[fwd_cols].values
#X = np.zeros((df.shape[0], 0))  # No macro variables
dates = df.index.to_numpy()

# === R² OOS function ===
def R2OOS(y_true, y_forecast):
    y_true = np.array(y_true)
    y_forecast = np.array(y_forecast)
    y_condmean = np.divide(y_true.cumsum(), (np.arange(y_true.size)+1))

    y_condmean = np.insert(y_condmean, 0, np.nan)[:-1]

    y_condmean[np.isnan(y_forecast)] = np.nan
    
    SSres = np.nansum((y_true - y_forecast)**2)
    SStot = np.nansum((y_true - y_condmean)**2)
    return 1 - SSres / SStot

# === Run Expanding Window Forecast from 1990-01 ===
n_obs = Y.shape[0]
n_maturities = Y.shape[1]

# Determine first forecast index corresponding to 1990-01-01
start_oos_idx = df.index.get_loc(pd.to_datetime(OOS_START_DATE))

y_preds = np.full_like(Y, np.nan)
print("Running Elastic Net with expanding window...\n")

for t in range(start_oos_idx, n_obs):
    y_preds[t, :] = ElasticNet_NoMacro_Plain(Xexog[:t+1, :], Y[:t+1, :])
    print(f"Forecast {t+1 - start_oos_idx}/{n_obs - start_oos_idx} ({df.index[t].strftime('%Y-%m')})")
    

# === Compute and report results ===
print("\n=== Elastic Net OOS R² Results ===")
for i, col in enumerate(maturities):
    y_true = Y[start_oos_idx:, i]
    y_forecast = y_preds[start_oos_idx:, i]
    r2 = R2OOS(y_true, y_forecast)
    print(f"{col}: {r2:.4f}")
