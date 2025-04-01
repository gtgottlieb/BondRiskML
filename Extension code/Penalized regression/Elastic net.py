import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit


def compute_oos_r2(y_true, y_pred, expanding_mean):
    """
    Compute out-of-sample R² using an expanding mean as a benchmark.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_mean = np.array(expanding_mean)

    ss_res = np.nansum((y_true - y_pred) ** 2)
    ss_tot = np.nansum((y_true - y_mean) ** 2)
    return 1 - ss_res / ss_tot


def calculate_expanding_mean_df(Y, dates, maturities, start_date):
    """
    Calculate a DataFrame holding the expanding mean for each target maturity.
    """
    n_obs = Y.shape[0]
    expanding_means = np.full_like(Y, np.nan, dtype=np.float64)
    
    for t in range(n_obs):
        expanding_means[t, :] = np.mean(Y[:t + 1, :], axis=0)
        
    df_expanding = pd.DataFrame(expanding_means, columns=maturities, index=dates)
    df_expanding = df_expanding.reset_index().rename(columns={'index': 'date'})
    df_expanding = df_expanding[df_expanding["date"] >= pd.to_datetime(start_date)]
    return df_expanding


def forecast_with_elasticnet(X_train, Y_train):
    """
    Run ElasticNetCV on the training data to forecast the target variable.
    Uses an expanding training window and reserves the last sample as test.
    """
    # Split latest observation for testing
    X_train_features = X_train[:-1, :]
    X_test = X_train[-1, :].reshape(1, -1)
    Y_train_target = Y_train[:-1, :]

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test)

    # Create a validation split: last 15% used for validation
    n_train = X_train_scaled.shape[0]
    n_val = int(np.round(n_train * 0.15))
    n_tr = n_train - n_val
    validation_fold = np.concatenate((np.full(n_tr, -1), np.full(n_val, 0)))
    ps = PredefinedSplit(validation_fold.tolist())

    n_targets = Y_train_target.shape[1]
    predictions = np.full((1, n_targets), np.nan)
    
    for i in range(n_targets):
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            max_iter=5000,
            cv=ps,
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train_scaled, Y_train_target[:, i])
        predictions[0, i] = model.predict(X_test_scaled)[0]
        
    return predictions


# === SETTINGS ===
EXCESS_RETURN_PATH = "data-folder/Fwd rates and xr/xr.xlsx"
FWD_RATE_PATH = "data-folder/Fwd rates and xr/forward_rates.xlsx"
OOS_START_DATE = "1990-01-01"
OOS_END_DATE = "2018-12-01"
FULL_START_DATE = "1971-08-01"  # For full sample filtering

# === Load and preprocess data ===
df_xr = pd.read_excel(EXCESS_RETURN_PATH)
df_fwd = pd.read_excel(FWD_RATE_PATH)

# Standardize column names and convert dates
df_xr.columns = [col.strip().lower() for col in df_xr.columns]
df_fwd.columns = [col.strip().lower() for col in df_fwd.columns]

df_xr['date'] = pd.to_datetime(df_xr['date'])
df_fwd['date'] = pd.to_datetime(df_fwd['date'])

# Merge datasets on 'date'
df = pd.merge(df_xr, df_fwd, on="date", suffixes=("_xr", "_fwd"))
df.set_index("date", inplace=True)

# Define target maturities and corresponding column names
target_maturities = ["2 y", "3 y", "4 y", "5 y", "7 y", "10 y"]
fwd_maturities = ["2 y", "3 y", "4 y", "5 y", "7 y", "10 y"]

target_cols = [maturity + "_xr" for maturity in target_maturities]
fwd_cols = [maturity + "_fwd" for maturity in fwd_maturities]

# Filter the dataset for the selected period
df = df[(df.index >= FULL_START_DATE) & (df.index <= OOS_END_DATE)].copy()

# === Prepare arrays for forecasting ===
Y = df[target_cols].to_numpy()
Xexog = df[fwd_cols].to_numpy()
dates = df.index.to_numpy()

# Calculate expanding means for benchmarks from the out-of-sample start date
expanding_means_df = calculate_expanding_mean_df(Y, df.index, target_maturities, OOS_START_DATE)
print(expanding_means_df.head())

# Identify the start index of the out-of-sample period
oos_start_idx = df.index.get_loc(pd.to_datetime(OOS_START_DATE))
n_obs = Y.shape[0]
y_preds = np.full_like(Y, np.nan)

print("Running Elastic Net with expanding window...\n")

# === Run expanding window forecasts ===
for t in range(oos_start_idx, n_obs):
    y_preds[t, :] = forecast_with_elasticnet(Xexog[:t + 1, :], Y[:t + 1, :])
    forecast_num = t - oos_start_idx + 1
    total_forecasts = n_obs - oos_start_idx
    print(f"Forecast {forecast_num}/{total_forecasts} ({df.index[t].strftime('%Y-%m')})")

# === Compute and report out-of-sample R² results ===
print("\n=== Elastic Net OOS R² Results ===")
for i, maturity in enumerate(target_maturities):
    y_true = Y[oos_start_idx:, i]
    y_forecast = y_preds[oos_start_idx:, i]
    benchmark_mean = expanding_means_df[maturity].iloc[:len(y_true)].values
    r2_score = compute_oos_r2(y_true, y_forecast, benchmark_mean)
    print(f"{maturity}: {r2_score:.4f}")