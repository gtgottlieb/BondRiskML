import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.optimizers import SGD

# Set seeds for replication
tf.random.set_seed(777)
np.random.seed(777)

# Parameters (same as in NN3_32_16_8.py)
differencing = True
pca_as_input = True
re_estimation_freq = 1  # in months
extended_sample_period = True
oos_start_date = '1990-01-01'
reestimation_start_date = '1991-01-01'
dumploc = 'data-folder\\dumploc_NN3_32_16_8'

# Import data
forward_rates = pd.read_excel("data-folder/!Data for forecasting/forward_rates.xlsx")
xr = pd.read_excel("data-folder/!Data for forecasting/xr.xlsx")
macro_df = pd.read_excel("data-folder/!Data for forecasting/Imputted_MacroData.xlsx")

# Set sample period
start_date = '1971-08-01'
if extended_sample_period:
    end_date = '2023-11-01'
else:
    end_date = '2018-12-01'

# Adjust dates in xr
xr['Date'] = pd.to_datetime(xr['Date']) - pd.DateOffset(years=1)

# Filter data by date
fwd_df = forward_rates[(forward_rates['Date'] >= start_date) & (forward_rates['Date'] <= end_date)]
xr_df = xr[(xr['Date'] >= start_date) & (xr['Date'] <= end_date)]
macro_df = macro_df[(macro_df['Date'] >= start_date) & (macro_df['Date'] <= end_date)]

if differencing:
    fwd_df = fwd_df.diff(12)
    xr_df = xr_df.shift(-12)
    macro_df = macro_df.shift(-12)
    combined = pd.concat([fwd_df, xr_df, macro_df], axis=1)
    combined.dropna(inplace=True)
    num_fwd_cols = fwd_df.shape[1]
    num_xr_cols = xr_df.shape[1]
    fwd_df = combined.iloc[:, :num_fwd_cols]
    xr_df = combined.iloc[:, num_fwd_cols:num_fwd_cols + num_xr_cols]
    macro_df = combined.iloc[:, num_fwd_cols + num_xr_cols:]

# Set Date as index
fwd_df.set_index('Date', inplace=True)
xr_df.set_index('Date', inplace=True)
macro_df.set_index('Date', inplace=True)

# Prepare X and Y variables
Y = xr_df.values
X_fwd = fwd_df.values
X_macro = macro_df.values
Y_index = xr_df.index

# Scale and, if required, apply PCA transformation
X_fwd_scaler = MinMaxScaler(feature_range=(-1,1))
X_macro_scaler = MinMaxScaler(feature_range=(-1,1))

if pca_as_input:
    pca_fwd = PCA(n_components=3)
    pca_macro = PCA(n_components=8)
    X_fwd = pca_fwd.fit_transform(X_fwd)
    X_macro = pca_macro.fit_transform(X_macro)

X_fwd_scaled = X_fwd_scaler.fit_transform(X_fwd)
X_macro_scaled = X_macro_scaler.fit_transform(X_macro)

# Function that loads the saved model and returns point prediction
def forecast_NN(X_f_test, X_m_test, model_no):
    model_path = os.path.join(dumploc, f'BestModel_{model_no}.keras')
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Skipping date {model_no}.")
        return None
    model = load_model(model_path)
    # Reshape test inputs as required (assumes one observation)
    X_f_test = X_f_test.reshape(1, -1)
    X_m_test = X_m_test.reshape(1, -1)
    Y_pred = model.predict([X_m_test, X_f_test])
    return Y_pred

# Generate out-of-sample iteration dates
oos_dates = pd.date_range(start=oos_start_date, end=end_date, freq=f'{re_estimation_freq}MS').normalize()

all_Y_pred = []
restimation_iteration_dates = []

# No training loop; simply loop over available dates and forecast if a saved model exists.
for t in oos_dates:
    if t < pd.to_datetime(reestimation_start_date):
        continue  # Skip dates before reestimation_start_date
    test_mask = (Y_index == t)
    if not any(test_mask):
        continue  # No test observation for this date
    X_f_test = X_fwd_scaled[test_mask]
    X_m_test = X_macro_scaled[test_mask]
    model_no = t.strftime('%Y%m')
    prediction = forecast_NN(X_f_test, X_m_test, model_no)
    if prediction is not None:
        all_Y_pred.append(prediction)
        restimation_iteration_dates.append(t)

if all_Y_pred:
    all_Y_pred = np.vstack(all_Y_pred)
    # Get maturity names from columns of xr_df (if available)
    maturity_names = xr_df.columns.tolist()
    Y_pred_df = pd.DataFrame(all_Y_pred, index=restimation_iteration_dates, columns=maturity_names)
    # Save the point predictions to an Excel file
    Y_pred_df.to_excel('data-folder/NN3_32_16_8_PointPredictions.xlsx')
    print("Point predictions saved to NN3_32_16_8_PointPredictions.xlsx")
else:
    print("No predictions were generated.")
