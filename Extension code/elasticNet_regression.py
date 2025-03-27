import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import mean_squared_error
import warnings

"""
This script replicates the Elastic Net model from Bianchi using the provided data.
It runs a full grid search to find optimal parameters (as Bianchi did not provide them) so it may take a while to complete.
"""
warnings.filterwarnings("ignore")

def prepare_data(xr_path, fwd_path):
    df_xr = pd.read_excel(xr_path)
    df_fwd = pd.read_excel(fwd_path)

    df_xr.columns = [c.strip().lower() for c in df_xr.columns]
    df_fwd.columns = [c.strip().lower() for c in df_fwd.columns]

    df_xr['date'] = pd.to_datetime(df_xr['date'])
    df_fwd['date'] = pd.to_datetime(df_fwd['date'])

    df = pd.merge(df_xr, df_fwd, on="date", suffixes=("_xr", "_fwd"))
    df.set_index("date", inplace=True)

    

    return df

def run_elastic_net(df, target_maturity, start_oos="1990-01-01"):
    target_col = f"{target_maturity} m_xr"
    fwd_cols = [col for col in df.columns if col.endswith("m_fwd")]

    # SHIFT TARGET FIRST
    df[target_col] = df[target_col].shift(-12)
    df = df[[target_col] + fwd_cols].dropna()

    oos_start_idx = df.index.get_loc(pd.to_datetime(start_oos))

    preds, rets, dates, nonzero_coefs = [], [], [], []

    for t in range(oos_start_idx, len(df) - 1):
        df_train = df.iloc[:t]
        df_test = df.iloc[[t]]

        X_train = df_train[fwd_cols].values
        y_train = df_train[target_col].values

        X_test = df_test[fwd_cols].values
        y_test = df_test[target_col].values

        # Standardize
        X_scaler = StandardScaler().fit(X_train)
        X_train_std = X_scaler.transform(X_train)
        X_test_std = X_scaler.transform(X_test)

        # Define PredefinedSplit (last 15% = validation)
        N = len(X_train_std)
        N_val = int(np.ceil(N * 0.15))
        N_train = N - N_val
        test_fold = np.concatenate((np.full(N_train, -1), np.full(N_val, 0)))
        ps = PredefinedSplit(test_fold)

        # Elastic NetCV with l1_ratio grid
        model = ElasticNetCV(cv=ps,
                             l1_ratio=[.1, .3, .5, .7, .9],
                             max_iter=5000,
                             n_jobs=-1,
                             random_state=42)
        model.fit(X_train_std, y_train)
        y_pred = model.predict(X_test_std)[0]

        preds.append(y_pred)
        rets.append(y_test[0])
        dates.append(df.index[t])
        nonzero_coefs.append(np.sum(model.coef_ != 0))

    # Compute R²_oos (Bianchi-style)
    preds = np.array(preds)
    rets = np.array(rets)
    # Construct lagged expanding mean (benchmark)
    rets = np.array(rets)
    preds = np.array(preds)

    # Shifted expanding mean benchmark
    benchmark = np.empty_like(rets)
    benchmark[0] = np.nan  # No history for first value

    for i in range(1, len(rets)):
        benchmark[i] = np.mean(rets[:i])

    # Mask out NaNs in both benchmark and predictions
    valid = ~np.isnan(benchmark) & ~np.isnan(preds)

    if np.sum(valid) < 10:
        print("Warning: too few valid data points to compute reliable R².")
        r2_oos = np.nan
    else:
        ss_res = np.nansum((rets[valid] - preds[valid])**2)
        ss_tot = np.nansum((rets[valid] - benchmark[valid])**2)
        r2_oos = 1 - ss_res / ss_tot


    # Print Diagnostics
    print(f"\nDiagnostics for {target_maturity}m:")
    print(f"→ Std. dev of actual returns:     {np.std(rets):.4f}")
    print(f"→ Std. dev of predicted returns:  {np.std(preds):.4f}")
    print(f"→ Model MSE:                      {np.mean(ss_res):.6f}")
    print(f"→ Benchmark MSE:                  {np.mean(ss_tot):.6f}")
    print(f"→ Avg. non-zero coefficients:     {np.mean(nonzero_coefs):.2f}")

    results = pd.DataFrame({
        "date": dates,
        "actual": rets,
        "predicted": preds,
        "benchmark_mean": benchmark,
        "nonzero_coefs": nonzero_coefs
    })

    return r2_oos, results

if __name__ == "__main__":
    # Input files
    excess_returns_file = "data-folder/Extracted_excess_returns.xlsx"
    forward_rates_file = "data-folder/Extracted_fwd_rates_new.xlsx"

    df_all = prepare_data(excess_returns_file, forward_rates_file)
    df_all = df_all[df_all.index <= "2018-12-01"]   # Same as Bianchi

    maturities = [24, 36, 48, 60, 84, 120]

    print("Elastic Net Results (OOS R²):\n")
    for mat in maturities:
        try:
            r2, df_results = run_elastic_net(df_all, mat)
            print(f"{mat}m: {r2:.4f}")
        except Exception as e:
            print(f"{mat}m: Error - {e}")
