import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import os
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import MultiTaskElasticNetCV


# === R² OOS function ===
def R2OOS(y_true, y_forecast, expanding_mean):
    y_true = np.array(y_true)
    y_forecast = np.array(y_forecast)
    y_mean = np.array(expanding_mean)

    SSres = np.nansum((y_true - y_forecast)**2)
    SStot = np.nansum((y_true - y_mean)**2)
    return 1 - SSres / SStot




# === SETTINGS ===
EXCESS_RETURN_PATH = "siebe/xrcorrect.xlsx"
FWD_RATE_PATH = "siebe/fwdcorrect.xlsx"
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

# Limit full data to training + OOS period
df = df[(df.index >= FULL_START_DATE) & (df.index <= OOS_END_DATE)].copy()

# === Prepare arrays ===
Y = df[target_cols].to_numpy()
Xexog = df[fwd_cols].to_numpy()
#X = np.zeros((df.shape[0], 0))  # No macro variables
dates = df.index.to_numpy()

# === Run Expanding Window Forecast from 1990-01 ===
n_obs = Y.shape[0]
n_maturities = Y.shape[1]

expanding_mean_check = np.full_like(Y, np.nan, dtype=np.float64)
for t in range(n_obs):
    expanding_mean_check[t, :] = Y[:t+1, :].mean(axis=0)

expanding_mean_df = pd.DataFrame(expanding_mean_check, columns=maturities, index=df.index)
expanding_mean_df = expanding_mean_df.iloc[220:]
print(expanding_mean_df.head())

# Determine first forecast index corresponding to 1990-01-01
start_oos_idx = df.index.get_loc(pd.to_datetime(OOS_START_DATE))

y_preds = np.full_like(Y, np.nan)
print("Running Elastic Net with expanding window...\n")


def ElasticNet_NoMacro_Plain(Xexog, Y):
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import PredefinedSplit
    from sklearn.model_selection import GridSearchCV

    Xexog_train = Xexog[:-1,:]
    Y_train = Y[:-1,:]
    Xexog_test = Xexog[-1,:]
    Xexog_test = Xexog_test.reshape(1, -1)

    Xexogscaler_train = StandardScaler()
    Xexog_train = Xexogscaler_train.fit_transform(Xexog_train)
    Xexog_test = Xexogscaler_train.transform(Xexog_test)

    # Construct Validation sample as last 15% of sample
    N_train = int(np.round(np.size(Xexog_train,axis=0)*0.85))
    N_val = np.size(Xexog_train,axis=0)-N_train
    test_fold =  np.concatenate(((np.full((N_train),-1),np.full((N_val),0))))
    ps = PredefinedSplit(test_fold.tolist())

    # Initialize prediction array
    Ypred = np.full([1, Y_train.shape[1]], np.nan)
    for i in range(Y_train.shape[1]):
        model = ElasticNetCV(max_iter=5000, n_jobs=-1, cv=ps, 
                             #l1_ratio = 1,
                             l1_ratio = [.1, .3, .5, .7, .9], 
                             #alphas = np.logspace(-3, 3, 50),  
                             random_state=42)
        model = model.fit(Xexog_train, Y_train[:,i])
        Ypred[0, i] = model.predict(Xexog_test)[0]
    return Ypred


#Create expanding mean and the predictions for each maturity
for t in range(start_oos_idx,n_obs):
    y_preds[t, :] = ElasticNet_NoMacro_Plain(Xexog[:t+1, :], Y[:t+1, :])
    print(f"Forecast {t+1 - start_oos_idx}/{n_obs - start_oos_idx} ({df.index[t].strftime('%Y-%m')})")

# === Compute and report results ===
print("\n=== Elastic Net OOS R² Results ===")
for i, col in enumerate(maturities):
    y_true = Y[start_oos_idx:, i]
    y_forecast = y_preds[start_oos_idx:, i]
    expanding_mean_for_maturity =  expanding_mean_df[col].iloc[0:len(y_true)].values
    r2 = R2OOS(y_true, y_forecast, expanding_mean_for_maturity)
    print(f"{col}: {r2:.4f}")

"""
# Plotting and exporting predictions
# === Save predictions to Excel ===
predictions_df = pd.DataFrame(y_preds, index=df.index, columns=maturities)
predictions_df = predictions_df.iloc[start_oos_idx:]  # only save OOS forecasts
output_file = "elastic_net_predictions.xlsx"
predictions_df.to_excel(output_file)
print(f"\nPredictions saved to {output_file}")

import matplotlib.pyplot as plt

# === Plot predictions vs true values ===
for i, col in enumerate(maturities):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index[start_oos_idx:], Y[start_oos_idx:, i], label="True", linewidth=1.5)
    plt.plot(df.index[start_oos_idx:], y_preds[start_oos_idx:, i], label="Predicted", linestyle="--")
    plt.title(f"Elastic Net Forecast vs Actual - {col}")
    plt.xlabel("Date")
    plt.ylabel("Excess Return")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()
"""