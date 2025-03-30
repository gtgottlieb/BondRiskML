import pandas as pd
import numpy as np
import NNFuncBib as NFB  # import the module that has ElasticNet_Exog_Plain
from scipy.stats import t as tstat
import statsmodels.api as sm

##############################################################################
# 1) HELPER FUNCTIONS
##############################################################################

def R2OOS(y_true, y_forecast):
    """
    Computes the out-of-sample R^2 relative to an expanding-mean benchmark:
       R^2_oos = 1 - SSE(model) / SSE(benchmark)
    """
    # 1) Compute the "expanding mean" of actual returns up to each index.
    #    For date i, this is the mean of y_true[:i], i.e. everything before i
    #    Then we shift by 1 so that time i forecast does not peek at y_true[i].
    y_cum = np.cumsum(y_true)
    n = np.arange(1, len(y_true) + 1)
    y_condmean = y_cum / n  # same length as y_true
    # shift by 1 so row i uses average up to i-1
    y_condmean = np.insert(y_condmean, 0, np.nan)[:-1]

    # Align with forecast
    # We'll drop any points where forecast is NaN
    # The code below simply ensures shapes match and excludes NaNs.
    mask = ~np.isnan(y_forecast)
    y_true_valid = y_true[mask]
    y_forecast_valid = y_forecast[mask]
    y_condmean_valid = y_condmean[mask]

    if len(y_true_valid) < 2:
        return np.nan

    ss_res = np.sum((y_true_valid - y_forecast_valid)**2)
    ss_bench = np.sum((y_true_valid - y_condmean_valid)**2)
    return 1 - ss_res / ss_bench

def run_elastic_net_only(X, Xexog, Y, dates, oos_start="1990-01-01"):
    """
    Replicates Bianchi's expanding-window out-of-sample approach
    *only* for the ElasticNet_Exog_Plain model.

    Inputs:
      X:       shape (T x K) - macro variables (can be empty if no macros).
      Xexog:   shape (T x N) - forward rates, yields, etc.
      Y:       shape (T x M) - target returns for M maturities
      dates:   array-like of length T with datetimes
      oos_start (str): e.g. '1990-01-01' is the date to start OOS

    Returns:
      preds:          (T x M) matrix of one-step-ahead forecasts
                      (NaN before OOS start)
      r2oos_by_matur: (M,) array of out-of-sample R^2 for each maturity
    """
    # Make sure dates is datetime-like
    dates = pd.to_datetime(dates)
    # Figure out OOS start index
    tstart = np.argmax(dates >= pd.to_datetime(oos_start))
    T = len(dates)
    M = Y.shape[1]

    # Prepare space for forecasts
    preds = np.full((T, M), np.nan)

    # === Expanding Window ===
    for i in range(tstart, T):
        # Train on data up to index i (i.e. [0..i]), then forecast i
        # ElasticNet_Exog_Plain returns a 1 x M array
        y_pred_i = NFB.ElasticNet_Exog_Plain(
            X[:i+1, :],    # macro up to i
            Xexog[:i+1, :],# forward rates up to i
            Y[:i+1, :]     # returns up to i
        )
        preds[i, :] = y_pred_i

    # === Compute OOS R^2 for each maturity
    r2oos_by_matur = []
    for m in range(M):
        y_true_m = Y[:, m]
        y_pred_m = preds[:, m]

        # focus on OOS portion only
        y_true_m_oos = y_true_m[tstart:]
        y_pred_m_oos = y_pred_m[tstart:]
        r2 = R2OOS(y_true_m_oos, y_pred_m_oos)
        r2oos_by_matur.append(r2)

    return preds, np.array(r2oos_by_matur)

##############################################################################
# 2) MAIN SCRIPT: Load data, set up arrays, run the model
##############################################################################
if __name__ == "__main__":
    # ------------------------------------------------
    # 2.1) Read "Extracted_excess_returns.xlsx" => Y
    # Columns might be: [Date, 24 m, 36 m, 48 m, 60 m, 84 m, 120 m]
    # We'll rename to drop spaces:
    df_xr = pd.read_excel("data-folder\Extracted_excess_returns.xlsx")
    df_xr.columns = [c.strip() for c in df_xr.columns]
    df_xr.rename(columns={"Date": "date"}, inplace=True)
    df_xr.sort_values("date", inplace=True)

    # The maturity columns become Y:
    maturities = ["24 m", "36 m", "48 m", "60 m", "84 m", "120 m"]
    Y_cols = maturities  # same as above
    # We'll convert them to a numeric NxM array
    Y_array = df_xr[Y_cols].values
    date_array = df_xr["date"].values

    # ------------------------------------------------
    # 2.2) Read "Extracted_fwd_rates_new.xlsx" => Xexog
    # We assume columns are: [Date, 24 m, 36 m, 48 m, 60 m, 84 m, 120 m]
    df_fwd = pd.read_excel("data-folder\Extracted_fwd_rates_new.xlsx")
    df_fwd.columns = [c.strip() for c in df_fwd.columns]
    df_fwd.rename(columns={"Date": "date"}, inplace=True)
    df_fwd.sort_values("date", inplace=True)

    # For simplicity, let's do an *inner merge* on 'date'
    # so the two dataframes line up exactly
    df_merged = pd.merge(df_xr, df_fwd, on="date", suffixes=("_xr","_fwd"), how="inner")

    # Now define Y from _xr columns, Xexog from _fwd columns
    Y_cols_merged = [f"{m} _xr" for m in maturities]
    F_cols_merged = [f"{m} _fwd" for m in maturities]
    Y_array = df_merged[Y_cols_merged].values
    Xexog_array = df_merged[F_cols_merged].values
    date_array = df_merged["date"].values

    # If you have no macro variables, we can define X as an empty array
    X_array = np.empty((len(df_merged), 0))  # shape (T, 0)

    # ------------------------------------------------
    # 2.3) Run the purely-ElasticNet approach
    #      Out-of-sample start is set to 1990-01-01 by default
    preds, r2oos = run_elastic_net_only(X_array, Xexog_array, Y_array, date_array)

    print("\nElastic Net OOS R^2 by maturity:")
    for m, val in zip(maturities, r2oos):
        print(f"  {m}: {val:.4f}")

    # Optionally: Save preds, etc.
    # df_merged has T rows; we can store predictions if you want
    for i,m in enumerate(maturities):
        df_merged[f"pred_{m}"] = preds[:, i]

    df_merged.to_excel("ElasticNet_OOS_Preds.xlsx", index=False)
    print("\nPredictions and data saved to 'ElasticNet_OOS_Preds.xlsx'.")
