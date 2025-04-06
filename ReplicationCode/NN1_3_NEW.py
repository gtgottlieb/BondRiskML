import pandas as pd
import numpy as np
#import Data_preprocessing as data_prep
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD
import Roos
import compute_benchmark
import ModelComparison_Rolling

## Upload and allign data

# Import yield and macro data, set in dataframe format with 'Date' column 
#forward_rates, xr = data_prep.process_yield_data(yields_df)
forward_rates = pd.read_excel("data-folder/Fwd rates and xr/forward_rates.xlsx", engine='openpyxl')
xr = pd.read_excel("data-folder/Fwd rates and xr/xr.xlsx", engine='openpyxl')

fwd_df, xr_df = pd.DataFrame(forward_rates), pd.DataFrame(xr)

macro_df = pd.read_excel("data-folder/Cleaned data/Yields+Final/Imputted_MacroData.xlsx", engine='openpyxl')
macro_df = macro_df.drop(index=0) # Drop "Transform" row
macro_df = macro_df.rename(columns={'sasdate':'Date'})
macro_df['Date'] = pd.to_datetime(macro_df['Date'])
macro_df = macro_df.interpolate(method='linear') # Interpolate missing values

# Set sample period as in Bianchi, later expand to end_date = '2023-11-01'
start_date = '1971-09-01' 
end_date = '2018-12-01'

fwd_df = fwd_df[(fwd_df['Date'] >= start_date) & (fwd_df['Date'] <= end_date)]
xr_df = xr_df[(xr_df['Date'] >= start_date) & (xr_df['Date'] <= end_date)]
macro_df = macro_df[(macro_df['Date'] >= start_date) & (macro_df['Date'] <= end_date)]

## Prepare variables
Y = xr_df.drop(columns = 'Date').values
X = fwd_df.drop(columns = 'Date').values
X_exog = macro_df.drop(columns = 'Date').values

# Scale variables
X_scaler = MinMaxScaler(feature_range=(-1,1))
X_exog_scaler = MinMaxScaler(feature_range=(-1,1))

X = X_scaler.fit_transform(X)
X_exog = X_exog_scaler.fit_transform(X_exog)

# Determine in-sample / out-of-sample split
oos_start_index = int(len(Y) * 0.85)
T = int(X.shape[0])
oos_indeces = range(oos_start_index, T) # Reduce end of range for testing
M = Y.shape[1] # Number of maturities

## Set up and fit NN(1 layer, 3 neurons) model
all_Y_pred = [] # Storage dictionary for predicted excess returns
all_Y_test = []

for i in range(oos_start_index, T):

    # Split data into test and train
    X_train, X_test = X[:i], X[i:i+1]
    Y_train, Y_test = Y[:i], Y[i:i+1]

    # Build layers
    input_layer = Input(shape=(X_train.shape[1],))
    hidden_layer = Dense(3, activation='relu')(input_layer)  # 3 neurons in hidden layer
    output_layer = Dense(Y_train.shape[1], activation='linear')(hidden_layer) 

    # Set up model
    model = Model(inputs=input_layer, outputs=output_layer)
    sgd_optimizer = SGD(learning_rate=0.015, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd_optimizer, loss='mse')

    # Fit model and get predictions
    model.fit(X_train, Y_train, epochs=100, verbose=0)
    Y_pred = model.predict(X_test)

    all_Y_pred.append(Y_pred.flatten())
    all_Y_test.append(Y_test.flatten())

all_Y_pred = np.array(all_Y_pred)
all_Y_test = np.array(all_Y_test)

## Compute performance measures
xr_insample_df = xr_df.drop(columns='Date').iloc[:oos_start_index]
xr_oos_df = xr_df.drop(columns='Date').iloc[oos_start_index:T]

Y_bench = compute_benchmark.compute_benchmark_prediction(
    xr_insample=xr_insample_df, 
    xr_oos=xr_oos_df
)
Y_bench = np.array(Y_bench)
all_R2_oos = {}

print("Calculation based on Campbell and Thompson (2008)")
for maturity in range(Y.shape[1]):
    r2_oos = Roos.r2_oos(actual=all_Y_test[:, maturity], predicted=all_Y_pred[:, maturity], benchmark=Y_bench[:, maturity])
    all_R2_oos[f"Maturity {maturity + 1}"] = r2_oos
    print(f"OOS R^2 Score for Maturity {maturity + 1}: {r2_oos:.4f}")

all_R2_oos = {}

"""
print("Package calculation")
for maturity in range(Y.shape[1]):
    r2_oos = ModelComparison_Rolling.R2OOS(y_true=all_Y_test[:, maturity], y_forecast=all_Y_pred[:, maturity])
    all_R2_oos[f"Maturity {maturity + 1}"] = r2_oos
    print(f"OOS R^2 Score for Maturity {maturity + 1}: {r2_oos:.4f}")
"""
