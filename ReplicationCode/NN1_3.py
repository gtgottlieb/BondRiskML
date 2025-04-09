import pandas as pd
import numpy as np
#import Data_preprocessing as data_prep
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import ModelComparison_Rolling
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.optimizers import SGD
from keras.models import load_model
from keras.regularizers import l1_l2
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import ParameterGrid


## Upload and allign data

# Import yield and macro data, set in dataframe format with 'Date' column 
fwd_df = pd.read_excel("data-folder/Fwd rates and xr/forward_rates.xlsx", engine='openpyxl')
xr_df = pd.read_excel("data-folder/Fwd rates and xr/xr.xlsx", engine='openpyxl')

# Set sample period
start_date = '1971-09-01' 
end_date = '2018-12-01' # As in Bianchi
# end_date = '2018-12-01' # Extended

fwd_df = fwd_df[(fwd_df['Date'] >= start_date) & (fwd_df['Date'] <= end_date)]
xr_df = xr_df[(xr_df['Date'] >= start_date) & (xr_df['Date'] <= end_date)]

## Prepare variables
Y = xr_df.drop(columns = 'Date').values
X = fwd_df.drop(columns = 'Date').values

# Scale variables
X_scaler = MinMaxScaler(feature_range=(-1,1))
X_scaled = X_scaler.fit_transform(X)

# Determine in-sample / out-of-sample split
oos_start_index = int(len(Y) * 0.85)
T = int(Y.shape[0])

## Set up and fit NN(1 layer, 3 neurons) model

# Change: Extend in-sample with predicted observation @ t+1, instead of actual value

def NN1_3(X, Y, no, l1l2, n_epochs=100):
    
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
    dumploc = './trainingDumps_'
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

re_estimation_freq = 1 # Re-estimation frequency for NN, in months
oos_iteration_indeces = range(oos_start_index, T, re_estimation_freq)
total_iterations = len(ParameterGrid(param_grid)) * (len(oos_iteration_indeces))
iteration_count = 0

all_Y_pred = []

for i in oos_iteration_indeces:
    best_score = float('inf')
    best_params = None

    for params in ParameterGrid(param_grid):
        
        iteration_count += 1
        print(f"Running iteration {iteration_count}/{total_iterations} | Testing params: {params}") 
        Y_pred_val, val_loss = NN1_3(X = X_scaled[:i,:], Y = Y[:i,:], no=i, **params)  

        if val_loss < best_score:
            best_score = val_loss
            best_params = params
        
    Y_pred, val_loss_final = NN1_3(X = X_scaled[:i,:], Y = Y[:i,:], no=T, **best_params) 
    all_Y_pred.append(Y_pred)  
   
all_Y_pred = np.vstack(all_Y_pred) 

# Analyze model performance
Y_test = Y[oos_start_index::re_estimation_freq,:]

for maturity in range(1,Y.shape[1]):
    r2_oos = ModelComparison_Rolling.R2OOS(y_true=Y_test[:, maturity], y_forecast=all_Y_pred[:, maturity])
    print(f"OOS R^2 Score for Maturity {maturity + 1}: {r2_oos:.4f}")

    r2_significance = ModelComparison_Rolling.RSZ_Signif(y_true=Y_test[:, maturity], y_forecast=all_Y_pred[:, maturity])
    print(f"R^2 Significance for Maturity {maturity + 1}: {r2_significance:.4f}")
