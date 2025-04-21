## Import libraries

# Basic libraries
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import random

# Scikit-learn + tensorflow + keras libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l1_l2

from keras.optimizers import SGD

# Custom libraries
import HAC_CW_adj_R2_signif_test
from Roos import r2_oos

## Set seeds for replication
tf.random.set_seed(777)
np.random.seed(777)
random.seed(777)

## Model setup : taking first differences and/or PCA as input (instead of fwd rates directly), re-estimation frequency

differencing = True
pca_as_input = True
re_estimation_freq = 1 # In months
extended_sample_period = True
epochs = 10

## Import + prep the data

# Import yield data, set in dataframe format.
xr = pd.read_excel("data-folder/!Data for forecasting/xr.xlsx")
forward_rates = pd.read_excel("data-folder/!Data for forecasting/forward_rates.xlsx")
fwd_df, xr_df = pd.DataFrame(forward_rates), pd.DataFrame(xr)

# Set sample period
start_date = '1971-08-01' 
if extended_sample_period:
    end_date = '2023-11-01' # Most recent
else:
    end_date = '2018-12-01' # As in Bianchi

#fwd_df = fwd_df[(fwd_df['Date'] >= start_date) & (fwd_df['Date'] <= end_date)]
#xr_df = xr_df[(xr_df['Date'] >= start_date) & (xr_df['Date'] <= end_date)]

# Filter excess returns to only include specific maturities
selected_maturities = ["2 y", "3 y", "4 y", "5 y", "7 y", "10 y"]
xr_df = xr_df[selected_maturities]

oos_start_date = '1990-01-01'
reestimation_start_date = '1991-01-01'
reestimation_start_index = fwd_df[fwd_df['Date'] == reestimation_start_date].index[0]

if differencing:
    fwd_df = fwd_df.diff(12)
    xr_df = xr_df.shift(-12)  # Shift Y to match the X diff at t

    # Drop the first row where diff is NaN
    valid_index = fwd_df.index[12:]
    fwd_df = fwd_df.loc[valid_index]
    xr_df = xr_df.loc[valid_index]

## Prepare X + Y variables
fwd_df.set_index('Date', inplace=True)
xr_df.set_index('Date', inplace=True)

Y, X = xr_df.values, fwd_df.values
Y_index = xr_df.index 

X_scaler = MinMaxScaler(feature_range=(-1, 1))

if pca_as_input:
    pca = PCA(n_components=3)
    X = pca.fit_transform(X)
    
X_scaled = X_scaler.fit_transform(X)

## Set up and fit NN(1 layer, 3 neurons) model

def train_NN(X_train, Y_train, model_no, l1l2, dropout_rate, n_epochs=epochs):
    Y_train = np.expand_dims(Y_train, axis=1)

    input_layer = Input(shape=(X_train.shape[1],))

    hidden_layer_1 = Dense(3, activation='relu', kernel_regularizer=l1_l2(l1l2))(input_layer)
    batch_norm_1 = BatchNormalization()(hidden_layer_1)
    drop_1 = Dropout(dropout_rate)(batch_norm_1)

    output_layer = Dense(Y_train.shape[2], activation='linear')(drop_1)

    model = Model(inputs=input_layer, outputs=output_layer)
    sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    
    dumploc = '/Users/gtgot/Desktop/dumploc_NN1_3'
    mcp = ModelCheckpoint(dumploc + f'/BestModel_{model_no}.keras', monitor='val_loss', save_best_only=False)

    model.compile(optimizer=sgd_optimizer, loss='mse')
    history = model.fit(X_train, Y_train, epochs=n_epochs, validation_split=0.15, shuffle=True, callbacks=[mcp], verbose=0)

    val_loss = min(history.history['val_loss'])
    model.save(dumploc + f'/BestModel_{model_no}.keras')

    return val_loss


def forecast_NN(X_test, model_no):
    dumploc = '/Users/gtgot/Desktop/dumploc_NN1_3'
    model = load_model(dumploc + f'/BestModel_{model_no}.keras')
    X_test = X_test.reshape(1, -1)
    Y_pred = model.predict(X_test)

    return Y_pred

## Generate predictions

# Set up hyperparameter grid
param_grid = {
    'l1l2': [0.5, 1],
    'dropout_rate': [0.1, 0.3, 0.5],
}

# Determine in/out-of-sample split
T = int(Y.shape[0])
oos_iteration_dates = pd.date_range(start=oos_start_date, 
                                     end=end_date, 
                                     freq=f'{re_estimation_freq}MS').normalize()

restimation_iteration_dates = []

# Set up loop, counters + storage for grid search
total_iterations = len(ParameterGrid(param_grid)) * (len(oos_iteration_dates))
iteration_count = 0
start_time = time.time()

all_Y_pred = []

for t in oos_iteration_dates:
    best_score = float('inf')
    best_params = None
    best_Y_pred = None

    for params in ParameterGrid(param_grid):
        
        iteration_count += 1
        print(f"Running iteration {iteration_count}/{total_iterations} | Testing params: {params}") 

        X_train, Y_train = X_scaled[Y_index < t], Y[Y_index < t]
        X_test = X_scaled[Y_index == t][0]

        val_loss = train_NN(X_train, Y_train, model_no=t.strftime('%Y%m'), l1l2=params['l1l2'], dropout_rate=params['dropout_rate'])

        if t >= pd.to_datetime(reestimation_start_date):
            Y_pred = forecast_NN(X_test, model_no=t.strftime('%Y%m'))
        else:
            Y_pred = None

        if val_loss < best_score:
            best_score = val_loss
            best_params = params
            best_Y_pred = Y_pred

    if best_Y_pred is not None and t >= pd.to_datetime(reestimation_start_date):
            all_Y_pred.append(best_Y_pred)
            restimation_iteration_dates.append(t)
   
all_Y_pred = np.vstack(all_Y_pred) 

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

#if first_differences and pca_as_input:
    print("\n=== Neural Network (1 layer, 3 neurons) OOS Performance (First Differences + PCA) ===")
#elif first_differences:
    print("\n=== Neural Network (1 layer, 3 neurons) OOS Performance (First Differences) ===")
if pca_as_input:
    print("\n=== Neural Network (1 layer, 3 neurons) OOS Performance (PCA) ===")
else:
    print("\n=== Neural Network (1 layer, 3 neurons) OOS Performance ===")

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
    
    plt.savefig(f'/Users/avril/Desktop/Seminar/Python Code/Plots/NN1_Forecast_vs_True_{clean_name}.png', dpi=300)
    plt.close()
    '''

# Compute total runtime
end_time = time.time()
total_runtime = end_time - start_time
mins, secs = divmod(total_runtime, 60)
print(f"\n Total runtime: {int(mins)} min {secs:.0f} sec")

# Save forecasts to excel file
Y_oos_df = pd.DataFrame(all_Y_pred, index=restimation_iteration_dates, columns=maturity_names)
Y_oos_df.to_excel('/Users/gtgott/Desktop/NN1_3_Predictions.xlsx')
