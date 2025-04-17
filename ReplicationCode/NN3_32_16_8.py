## Import libraries

# Basic libraries
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# Scikit-learn + tensorflow + keras libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l1_l2

from keras.optimizers import SGD

# Custom libraries
import HAC_CW_adj_R2_signif_test
from Roos import r2_oos

## Set seeds for replication
tf.random.set_seed(777)
np.random.seed(777)

# Global boolean to control whether to check for existing models in dumploc
resume = False

## Model setup : taking first differences and/or PCA as input (instead of fwd rates directly), re-estimation frequency
differencing = False
pca_as_input = True
re_estimation_freq = 1 # In months
extended_sample_period = True
epochs = 50 

## Import + prep the data

# Import yield and macro data
forward_rates = pd.read_excel("data-folder/!Data for forecasting/forward_rates.xlsx")
xr = pd.read_excel("data-folder/!Data for forecasting/xr.xlsx")
fwd_df, xr_df = pd.DataFrame(forward_rates), pd.DataFrame(xr)

macro_df = pd.read_excel("data-folder/!Data for forecasting/Imputted_MacroData.xlsx") 


# Set sample period
start_date = '1971-08-01' 
if extended_sample_period:
    end_date = '2023-11-01' # Most recent
else:
    end_date = '2018-12-01' # As in Bianchi

# Shift the 'Date' column back by a year in the excess returns (xr_df)
xr_df['Date'] = pd.to_datetime(xr_df['Date']) - pd.DateOffset(years=1)

fwd_df = fwd_df[(fwd_df['Date'] >= start_date) & (fwd_df['Date'] <= end_date) ]
xr_df = xr_df[(xr_df['Date'] >= start_date) & (xr_df['Date'] <= end_date)]
macro_df = macro_df[(macro_df['Date'] >= start_date) & (macro_df['Date'] <= end_date) ]

oos_start_date = '1990-01-01'
reestimation_start_date = '1991-01-01'
reestimation_start_index = fwd_df[fwd_df['Date'] == reestimation_start_date].index[0]

if differencing:
    fwd_df = fwd_df.diff(12)

    # Drop NaNs across all datasets by merging and dropping rows with any missing values
    combined = pd.concat([fwd_df, xr_df, macro_df], axis=1)
    combined.dropna(inplace=True)

    # Split back into separate DataFrames
    num_fwd_cols = fwd_df.shape[1]
    num_xr_cols = xr_df.shape[1]
    fwd_df = combined.iloc[:, :num_fwd_cols]
    xr_df = combined.iloc[:, num_fwd_cols:num_fwd_cols + num_xr_cols]
    macro_df = combined.iloc[:, num_fwd_cols + num_xr_cols:]

## Prepare X + Y variables
fwd_df.set_index('Date', inplace=True)
xr_df.set_index('Date', inplace=True)
macro_df.set_index('Date', inplace=True)

Y, X_fwd, X_macro = xr_df.values, fwd_df.values, macro_df.values
Y_index = xr_df.index 

X_fwd_scaler = MinMaxScaler(feature_range=(-1,1))
X_macro_scaler = MinMaxScaler(feature_range=(-1,1))

if pca_as_input:
    pca_fwd, pca_macro = PCA(n_components=3), PCA(n_components=8)
    X_fwd, X_macro = pca_fwd.fit_transform(X_fwd), pca_macro.fit_transform(X_macro)

X_fwd_scaled, X_macro_scaled = X_fwd_scaler.fit_transform(X_fwd), X_macro_scaler.fit_transform(X_macro)

## Set up and fit NN(1 layer, 3 neurons) model

def train_NN(X_f_train, X_m_train, Y_train, model_no, l1l2, dropout_rate, n_epochs=epochs):

    Y_train = np.expand_dims(Y_train, axis=1)

    # Set up model architecture (3 hidden layers, 32-16-8 neurons per layer)
    f_input = Input(shape=(X_f_train.shape[1],))
    m_input = Input(shape=(X_m_train.shape[1],))

    hidden_layer_1 = Dense(32, activation='relu', kernel_regularizer=l1_l2(l1l2))(m_input)
    batch_norm_1 = BatchNormalization()(hidden_layer_1)
    drop_1 = Dropout(dropout_rate)(batch_norm_1)

    hidden_layer_2 = Dense(16, activation='relu', kernel_regularizer=l1_l2(l1l2))(drop_1)
    batch_norm_2 = BatchNormalization()(hidden_layer_2)
    drop_2 = Dropout(dropout_rate)(batch_norm_2)

    hidden_layer_3 = Dense(8, activation='relu', kernel_regularizer=l1_l2(l1l2))(drop_2)
    batch_norm_3 = BatchNormalization()(hidden_layer_3)
    drop_3 = Dropout(dropout_rate)(batch_norm_3)

    add_f = Concatenate()([drop_3, f_input])
    output_layer = Dense(10, activation='linear')(add_f) 
    
    model = Model(inputs=[m_input, f_input], outputs=output_layer)
    sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

    dumploc = 'data-folder\dumploc_NN3_32_16_8'  # Use a relative path in the current working directory
    mcp = ModelCheckpoint(dumploc + f'/BestModel_{model_no}.keras', monitor='val_loss', save_best_only=False)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    model.compile(optimizer=sgd_optimizer, loss='mse')
    history = model.fit([X_m_train, X_f_train], Y_train, epochs=n_epochs, callbacks=[mcp,early_stopping], 
                           shuffle=True, validation_split = 0.15, verbose=0, )
    
    val_loss = min(history.history['val_loss'])
    model.save(dumploc + f'/BestModel_{model_no}.keras')

    return val_loss

def forecast_NN(X_f_test, X_m_test, model_no):
    dumploc = 'data-folder\dumploc_NN3_32_16_8'  # Use a relative path in the current working directory
    model = load_model(dumploc + f'/BestModel_{model_no}.keras')
    
    X_f_test = X_f_test.reshape(1, -1)
    X_m_test = X_m_test.reshape(1, -1)
    
    Y_pred = model.predict([X_m_test, X_f_test])

    return Y_pred

# Generate predictions

# Set up hyperparameter grid
param_grid = {
    'l1l2': [0.01, 0.001],
    'dropout_rate': [0.1, 0.3, 0.5]
}

T = int(Y.shape[0])
oos_iteration_dates = pd.date_range(start=oos_start_date, 
                                     end=end_date, 
                                     freq=f'{re_estimation_freq}MS').normalize()

# --- New code for resume functionality ---
import os
if resume:
    pending_dates = [t for t in oos_iteration_dates 
                     if not os.path.exists(f'data-folder/dumploc_NN3_32_16_8/BestModel_{t.strftime("%Y%m")}.keras')]
else:
    pending_dates = list(oos_iteration_dates)

total_iterations = len(ParameterGrid(param_grid)) * (len(pending_dates))
iteration_count = 0
start_time = time.time()

all_Y_pred = []
restimation_iteration_dates = []

for t in pending_dates:
    # No need to check again inside the loop since pending_dates already excludes them
    best_val_loss = float('inf')
    best_params = None
    best_Y_pred = None

    for params in ParameterGrid(param_grid):
        iteration_count += 1
        print(f"Running iteration {iteration_count}/{total_iterations} | Testing params: {params}") 

        X_f_train, X_m_train, Y_train = X_fwd_scaled[Y_index < t], X_macro_scaled[Y_index < t], Y[Y_index < t]
        X_f_test, X_m_test = X_fwd_scaled[Y_index == t], X_macro_scaled[Y_index == t]

        val_loss = train_NN(X_f_train, X_m_train, Y_train, model_no=t.strftime('%Y%m'), **params)

        if t >= pd.to_datetime(reestimation_start_date):
            Y_pred = forecast_NN(X_f_test, X_m_test, model_no=t.strftime('%Y%m'))
        else:
            Y_pred = None

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
            best_Y_pred = Y_pred

    if best_Y_pred is not None and t >= pd.to_datetime(reestimation_start_date):
        all_Y_pred.append(best_Y_pred)
        restimation_iteration_dates.append(t)

all_Y_pred = np.vstack(all_Y_pred)  # All new forecasts are aggregated here

## Analyze model performance
Y_test = Y[Y_index.isin(restimation_iteration_dates)]

# Silence warning messages for .iloc in significance test (not relevant for now)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

maturity_names = xr_df.columns.tolist()

def compute_benchmark_prediction(xr_insample, xr_oos):
    benchmark_preds = []

    for i in range(len(xr_oos)):
        combined = np.concatenate([xr_insample, xr_oos[:i+1]]) 
        avg_val = combined.mean()  # This computes column-wise means
        benchmark_preds.append(avg_val)

    # Convert list of Series (one per iteration) to a single DataFrame
    return pd.DataFrame(benchmark_preds, index=np.arange(len(xr_oos)))

if differencing and pca_as_input:
    print("\n=== Neural Network (3 layers, 32-16-8 neurons per layer) OOS Performance (First Differences + PCA) ===")
elif differencing:
    print("\n=== Neural Network (3 layers, 32-16-8 neurons per layer) OOS Performance (First Differences) ===")
elif pca_as_input:
    print("\n=== Neural Network (3 layers, 32-16-8 neurons per layer) OOS Performance (PCA) ===")
else:
    print("\n=== Neural Network (3 layers, 32-16-8 neurons per layer) OOS Performance ===")

for maturity in range(1,Y.shape[1]):
    maturity_name = maturity_names[maturity]

    benchmark = compute_benchmark_prediction(Y[:reestimation_start_index, maturity],Y_test[:, maturity]).squeeze()
    r2_value = r2_oos(Y_test[:, maturity], all_Y_pred[:, maturity], benchmark[maturity])
    rmse = np.sqrt(np.mean((Y_test[:, maturity] - all_Y_pred[:, maturity])**2))
    signif_test_stat, signif_p_value = HAC_CW_adj_R2_signif_test.get_CW_adjusted_R2_signif(Y_test[:, maturity], all_Y_pred[:, maturity], benchmark)

    print(f"{maturity_name}: R²OOS={r2_value*100:.3f}%, RMSE={rmse:.3f}, p-value={signif_p_value:.3f}")

    '''
    plt.figure(figsize=(12, 6))
    plt.plot(restimation_iteration_dates, Y_test[:, maturity], label='True', color='red')
    plt.plot(restimation_iteration_dates, all_Y_pred[:, maturity], label='Forecast', color='blue', linestyle='--')
    plt.title(f'Forecast vs True | {maturity_name}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    clean_name = maturity_name.replace(" ", "_").replace("/", "_")
    
    plt.savefig(f'/Users/avril/Desktop/Seminar/Python Code/Plots/NN3_32_16_8_Forecast_vs_True_{clean_name}.png', dpi=300)
    plt.close()
    '''

# Compute total runtime
end_time = time.time()
total_runtime = end_time - start_time
mins, secs = divmod(total_runtime, 60)
print(f"\n Total runtime: {int(mins)} min {secs:.0f} sec")

# Save forecasts to excel file
# NOTE: This Excel file (NN3_32_16_8_Predictions.xlsx) will only contain forecasts produced in the current run.
Y_oos_df = pd.DataFrame(all_Y_pred, index=restimation_iteration_dates, columns=maturity_names)
Y_oos_df.to_excel('data-folder/NN3_32_16_8_Predictions.xlsx')
