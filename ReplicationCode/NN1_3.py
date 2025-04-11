import pandas as pd
import numpy as np
import Data_preprocessing as data_prep
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import ModelComparison_Rolling
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l1_l2
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import PCA
import time

'''
TO DO:
* 
'''

## Upload and allign data

first_differences = False
pca_as_input = True

# Import yield data, set in dataframe format with 'Date' column 
yields_df = pd.read_excel('/Users/avril/Desktop/Seminar/Data/Aligned_Yields_Extracted.xlsx')
forward_rates, xr = data_prep.process_yield_data(yields_df)
fwd_df, xr_df = pd.DataFrame(forward_rates), pd.DataFrame(xr)

# Set sample period
start_date = '1971-08-01' 
end_date = '2018-12-01' # As in Bianchi

fwd_df = fwd_df[(fwd_df['Date'] >= start_date) & (fwd_df['Date'] <= end_date)]
xr_df = xr_df[(xr_df['Date'] >= start_date) & (xr_df['Date'] <= end_date)]

oos_start_date = '1990-01-01'
oos_start_index = fwd_df[fwd_df['Date'] == oos_start_date].index[0]

if first_differences:
    fwd_df = fwd_df.diff()
    xr_df = xr_df.shift(-1)  # Shift Y to match the X diff at t

    # Drop the first row where diff is NaN
    valid_index = fwd_df.index[1:]
    fwd_df = fwd_df.loc[valid_index]
    xr_df = xr_df.loc[valid_index]

## Prepare variables
xr_df, fwd_df = xr_df.drop(columns = 'Date'),fwd_df.drop(columns = 'Date')
Y, X = xr_df.values, fwd_df.values

X_scaler = MinMaxScaler(feature_range=(-1, 1))

if pca_as_input:
    pca = PCA(n_components=3)
    X = pca.fit_transform(X)
    
X_scaled = X_scaler.fit_transform(X)

## Set up and fit NN(1 layer, 3 neurons) model

def NN1_3(X, Y, no, l1l2, n_epochs=10):
    
    X_train, Y_train = X[:-1,:], Y[:-1,:] 
    X_test = X[-1,:].reshape(1,-1)
    Y_train = np.expand_dims(Y_train, axis=1)

    # Set seeds for replication
    tf.random.set_seed(777)
    np.random.seed(777)

    # Set up model architecture (1 hidden layer, 3 neurons)
    input_layer = Input(shape=(X_train.shape[1],))
    hidden_layer = Dense(3, activation='relu', kernel_regularizer=l1_l2(l1l2))(input_layer)
    output_layer = Dense(Y_train.shape[2], activation='linear')(hidden_layer) 
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile model
    sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    dumploc = '/Users/avril/Desktop/Seminar/Python Code/dumploc_NN1_3'
    mcp = ModelCheckpoint(dumploc+'/BestModel_'+str(no)+'.keras',
                              monitor='val_loss',save_best_only=False)
    model.compile(optimizer=sgd_optimizer, loss='mse')

    model_hist = model.fit(X_train, Y_train, epochs=n_epochs, callbacks=[mcp], 
                           shuffle=True, validation_split = 0.15, verbose=0, )
    
    model = load_model(dumploc+'/BestModel_'+str(no)+'.keras')
    model.save(dumploc+'/BestModel_'+str(no)+'.keras')

    X_scaled_test = X_scaler.fit_transform(X_test)

    Y_pred = model.predict(X_scaled_test)
    val_loss = min(model_hist.history['val_loss'])

    return Y_pred, val_loss

# Generate predictions

param_grid = {
    'l1l2': [0.01, 0.001]
}

# Determine train-validation split
T = int(Y.shape[0])
re_estimation_freq = 3 # Re-estimation frequency for NN, in months
oos_iteration_indeces = [oos_start_index] + list(range(oos_start_index + 11, T, re_estimation_freq))
total_iterations = len(ParameterGrid(param_grid)) * (len(oos_iteration_indeces))
iteration_count = 0
all_Y_pred = []
start_time = time.time()

for i in oos_iteration_indeces:
    best_score = float('inf')
    best_params = None
    best_Y_pred = None

    for params in ParameterGrid(param_grid):
        
        iteration_count += 1
        print(f"Running iteration {iteration_count}/{total_iterations} | Testing params: {params}") 
        Y_pred, val_loss = NN1_3(X = X_scaled[:i,:], Y = Y[:i,:], no=i, **params)  

        if val_loss < best_score:
            best_score = val_loss
            best_params = params
            best_Y_pred = Y_pred
        
    all_Y_pred.append(Y_pred)  
   
all_Y_pred = np.vstack(all_Y_pred) 

# Analyze model performance
Y_test = Y[oos_iteration_indeces,:]

import HAC_CW_adj_R2_signif_test
import compute_benchmark
benchmark = compute_benchmark.compute_benchmark_prediction(Y[:oos_start_index-1,:], Y_test)
benchmark = benchmark.values.ravel()
maturity_names = xr_df.columns.tolist()

if first_differences:
    print("\n=== Neural Network (1 layer, 3 neurons) OOS Performance (First Differences) ===")
else:
    print("\n=== Neural Network (1 layer, 3 neurons) OOS Performance ===")

for maturity in range(1,Y.shape[1]):
    maturity_name = maturity_names[maturity]
    r2_oos = ModelComparison_Rolling.R2OOS(y_true=Y_test[:, maturity], y_forecast=all_Y_pred[:, maturity])
    mspe = np.mean((Y_test[:, maturity] - all_Y_pred[:, maturity]) ** 2)
    signif_test_stat, signif_p_value = HAC_CW_adj_R2_signif_test.get_CW_adjusted_R2_signif(Y_test[:, maturity], all_Y_pred[:, maturity], benchmark)
    print(f"{maturity_name}: R²OOS={r2_oos:.3f}%, MSPE={mspe:.3f}, p-value={signif_p_value:.3f}")

# Compute total runtime
end_time = time.time()
total_runtime = end_time - start_time
mins, secs = divmod(total_runtime, 60)
print(f"\n Total runtime: {int(mins)} min {secs:.2f} sec")
