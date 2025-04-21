from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------------- #
#  Helper functions
# --------------------------------------------------------------------------- #

def compute_benchmark_prediction(
    xr_insample: np.ndarray, xr_oos: np.ndarray
) -> np.ndarray:
    """Expanding historical‑mean benchmark (vectorised)."""
    cum_sum = np.cumsum(np.r_[xr_insample, xr_oos])
    count = np.arange(1, cum_sum.size + 1)
    return cum_sum / count[len(xr_insample) :]


def r2_oos(y_true: np.ndarray, y_pred: np.ndarray, benchmark: np.ndarray) -> float:
    ss_pred = np.sum((y_true - y_pred) ** 2)
    ss_bench = np.sum((y_true - benchmark) ** 2)
    return 1.0 - ss_pred / ss_bench


def month_range(start: pd.Timestamp, end: pd.Timestamp, step: int) -> List[pd.Timestamp]:
    return list(pd.date_range(start=start, end=end, freq=f"{step}MS").normalize())


def align_on_intersection(df_x: pd.DataFrame, df_y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only dates present in *both* DataFrames."""
    common = pd.Series(np.intersect1d(df_x["Date"].values, df_y["Date"].values))
    df_x = df_x[df_x["Date"].isin(common)].reset_index(drop=True)
    df_y = df_y[df_y["Date"].isin(common)].reset_index(drop=True)
    return df_x, df_y

# --------------------------------------------------------------------------- #
#  Config flags (edit as needed)
# --------------------------------------------------------------------------- #
DIFFERENCING = True
PCA_AS_INPUT = True
PCA_COMPONENTS = 3
RE_ESTIMATION_FREQ = 1  # months
EXTENDED_SAMPLE = True
EPOCHS = 500
RANDOM_STATE = 777

PARAM_GRID = {
    "hidden_layer_sizes": [(3,), (3, 3)],
    "alpha": [1e-4, 1e-3],
}

# --------------------------------------------------------------------------- #
#  Load & filter data
# --------------------------------------------------------------------------- #
DATA_DIR = Path("data-folder/!Data for forecasting")

xr_df = pd.read_excel(DATA_DIR / "xr.xlsx")
fwd_df = pd.read_excel(DATA_DIR / "forward_rates.xlsx")

for df in (xr_df, fwd_df):
    df["Date"] = pd.to_datetime(df["Date"])

START_DATE = pd.Timestamp("1971-08-01")
END_DATE = pd.Timestamp("2023-11-01") if EXTENDED_SAMPLE else pd.Timestamp("2018-12-01")

# filter each DataFrame independently, then intersect to guarantee alignment
#xr_df = xr_df[(xr_df["Date"] >= START_DATE) & (xr_df["Date"] <= END_DATE)].copy()
#fwd_df = fwd_df[(fwd_df["Date"] >= START_DATE) & (fwd_df["Date"] <= END_DATE)].copy()

# Make absolutely sure dates match (fixes the IndexingError)
xr_df, fwd_df = align_on_intersection(xr_df, fwd_df)

# --------------------------------------------------------------------------- #
#  Differencing & alignment of Y
# --------------------------------------------------------------------------- #
if DIFFERENCING:
    diff_m = 12
    fwd_df.iloc[:, 1:] = fwd_df.iloc[:, 1:].diff(diff_m)
    xr_df.iloc[:, 1:] = xr_df.iloc[:, 1:].shift(-diff_m)
    valid = fwd_df.index[diff_m:]
    fwd_df = fwd_df.loc[valid].reset_index(drop=True)
    xr_df = xr_df.loc[valid].reset_index(drop=True)

# align again after differencing (defensive)
xr_df, fwd_df = align_on_intersection(xr_df, fwd_df)

index = xr_df["Date"].copy()
X_raw = fwd_df.drop(columns="Date").values
Y = xr_df.drop(columns="Date").values
maturity_names = xr_df.columns.drop("Date").tolist()

# --------------------------------------------------------------------------- #
#  Pre‑processing pipeline
# --------------------------------------------------------------------------- #
steps = [("scaler", MinMaxScaler(feature_range=(-1, 1)))]
if PCA_AS_INPUT:
    steps.append(("pca", PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)))
pre_base = Pipeline(steps)

# --------------------------------------------------------------------------- #
#  OOS forecasting loop
# --------------------------------------------------------------------------- #
OOS_START_DATE = pd.Timestamp("1990-01-01")
RE_ESTIM_START = pd.Timestamp("1991-01-01")

oos_dates = month_range(OOS_START_DATE, END_DATE, RE_ESTIMATION_FREQ)

preds, pred_dates = [], []

warnings.filterwarnings("ignore", category=UserWarning)

for date in oos_dates:
    train = index < date
    test = index == date
    if not test.any():
        continue

    X_train, X_test = X_raw[train], X_raw[test]
    Y_train = Y[train]

    pre = pre_base.fit(X_train)
    X_train = pre.transform(X_train)
    X_test = pre.transform(X_test)

    best_loss, best_pred = np.inf, None
    for params in ParameterGrid(PARAM_GRID):
        mlp = MLPRegressor(
            activation="relu",
            solver="adam",
            learning_rate="adaptive",
            max_iter=EPOCHS,
            validation_fraction=0.15,
            early_stopping=True,
            n_iter_no_change=20,
            tol=1e-4,
            random_state=RANDOM_STATE,
            **params,
        )
        mlp.fit(X_train, Y_train)
        loss = mlp.loss_
        if loss < best_loss:
            best_loss = loss
            best_pred = mlp.predict(X_test)[0]
    if date >= RE_ESTIM_START:
        preds.append(best_pred)
        pred_dates.append(date)
    print(f"{date.date()} | best loss {best_loss:.5f}")

# --------------------------------------------------------------------------- #
#  Performance metrics
# --------------------------------------------------------------------------- #
Y_oos_true = Y[np.isin(index, pred_dates)]
preds = np.vstack(preds)
pred_index = pd.DatetimeIndex(pred_dates, name="Date")

benchmarks = []
for j, d in enumerate(pred_dates):
    idx_pred = np.where(index == d)[0][0]
    bench = compute_benchmark_prediction(Y[:idx_pred], Y_oos_true[j:j+1])
    benchmarks.append(bench)
benchmarks = np.vstack(benchmarks)

print("\n=== OOS performance ===")
for m, name in enumerate(maturity_names):
    r2 = r2_oos(Y_oos_true[:, m], preds[:, m], benchmarks[:, m])
    rmse = np.sqrt(mean_squared_error(Y_oos_true[:, m], preds[:, m]))
    print(f"{name:>5} | R² {r2*100:6.2f}% | RMSE {rmse:.5f}")

# Save forecasts
a_out = Path("nn_mlp_predictions.xlsx")
pd.DataFrame(preds, index=pred_index, columns=maturity_names).to_excel(a_out)
print(f"Saved forecasts to {a_out.resolve()}")
