import pandas as pd
import numpy as np

from pca_regression import main as run_pca
from elastic_net_2 import ElasticNet_NoMacro_Plain
from sklearn.metrics import mean_squared_error
from Roos import r2_oos

# === Paths ===
EXCESS_RETURN_PATH = "siebe/xrcorrect.xlsx"
FWD_RATE_PATH = "siebe/fwdcorrect.xlsx"
OOS_START_DATE = "1990-01-01"
OOS_END_DATE = "2018-12-01"
FULL_START_DATE = "1971-08-01"

# === Load Data ===
df_xr = pd.read_excel(EXCESS_RETURN_PATH)
df_fwd = pd.read_excel(FWD_RATE_PATH)
df_xr['date'] = pd.to_datetime(df_xr['date'])
df_fwd['date'] = pd.to_datetime(df_fwd['date'])

# Normalize column names
df_xr.columns = [c.strip().lower() for c in df_xr.columns]
df_fwd.columns = [c.strip().lower() for c in df_fwd.columns]

# Merge and filter data
df = pd.merge(df_xr, df_fwd, on="date", suffixes=("_xr", "_fwd"))
df.set_index("date", inplace=True)
df = df[(df.index >= FULL_START_DATE) & (df.index <= OOS_END_DATE)].copy()

# Define columns
maturities = ["24 m", "36 m", "48 m", "60 m", "84 m", "120 m"]
target_cols = [m + "_xr" for m in maturities]
fwd_cols = [m + "_fwd" for m in maturities]

# === Prepare arrays ===
Y = df[target_cols].to_numpy()
Xexog = df[fwd_cols].to_numpy()
dates = df.index.to_numpy()
n_obs = Y.shape[0]
n_maturities = Y.shape[1]

# Find OOS start index
start_oos_idx = df.index.get_loc(pd.to_datetime(OOS_START_DATE))

# === Run PCA forecast ===
print("Running PCA forecasts...")
from pca_regression import split_data_by_date, iterative_pca_regression

df_reset = df.reset_index()
df_reset.rename(columns={"date": "Date"}, inplace=True)

data_split = split_data_by_date(df_reset[target_cols + ["Date"]],
                                 df_reset[fwd_cols + ["Date"]],
                                 OOS_START_DATE, OOS_END_DATE)

er_in = data_split["excess_returns_in"]
er_out = data_split["excess_returns_out"]
fr_in = data_split["forward_rates_in"]
fr_out = data_split["forward_rates_out"]

pca_preds = {}
for i, col in enumerate(er_out.columns):
    pca_preds[col] = iterative_pca_regression(er_in[[col]], fr_in.copy(), er_out[[col]], fr_out.copy())
pca_preds_df = pd.DataFrame(pca_preds)

# === Run Elastic Net forecasts ===
print("Running Elastic Net forecasts...")
elastic_net_preds = np.full_like(Y, np.nan)
for t in range(start_oos_idx, n_obs):
    elastic_net_preds[t, :] = ElasticNet_NoMacro_Plain(Xexog[:t+1, :], Y[:t+1, :])
elastic_net_preds_df = pd.DataFrame(elastic_net_preds[start_oos_idx:],
                                     index=df.index[start_oos_idx:],
                                     columns=target_cols)

# === Align PCA preds to same dates/columns ===
pca_preds_df.index = df.index[start_oos_idx:start_oos_idx + len(pca_preds_df)]
pca_preds_df = pca_preds_df[target_cols]

# === True values ===
y_true_df = pd.DataFrame(Y[start_oos_idx:], index=df.index[start_oos_idx:], columns=target_cols)

# === Compute MSE and weights ===
def compute_mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

def compute_weights(y_true, pred_dict):
    mse = {model: compute_mse(y_true, preds) for model, preds in pred_dict.items()}
    inv_mse = {k: 1/v for k, v in mse.items()}
    total = sum(inv_mse.values())
    return {k: v/total for k, v in inv_mse.items()}

pred_dict = {
    "PCA": pca_preds_df,
    "ElasticNet": elastic_net_preds_df
}

weights = compute_weights(y_true_df, pred_dict)
print("Model Weights:", weights)

# === Weighted average forecast ===
combined_forecast = sum(pred_dict[k] * weights[k] for k in pred_dict)

# === Compute and print R² for each maturity ===
print("\nCombined Forecast R² (OOS):")
for col in target_cols:
    r2 = r2_oos(y_true_df[col], combined_forecast[col], y_true_df[col].expanding().mean())
    print(f"{col}: {r2:.4f}")

# Optionally export
combined_forecast.to_excel("combined_forecast.xlsx")
print("\nSaved combined forecast to 'combined_forecast.xlsx'")
