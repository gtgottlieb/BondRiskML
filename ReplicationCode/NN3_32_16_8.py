import pandas as pd
import numpy as np
import Data_preprocessing as data_prep
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import ModelComparison_Rolling
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Concatenate
from keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l1_l2
from sklearn.model_selection import ParameterGrid

## Upload and allign data

# Import yield and macro data, set in dataframe format with 'Date' column 
yields_df = pd.read_excel('/Users/avril/Desktop/Seminar/Data/Aligned_Yields_Extracted.xlsx')
forward_rates, xr = data_prep.process_yield_data(yields_df)
fwd_df, xr_df = pd.DataFrame(forward_rates), pd.DataFrame(xr)
macro_df = pd.read_excel('/Users/avril/Desktop/Seminar/Data/Imputted_MacroData.xlsx')

# Set sample period as in Bianchi, later expand to end_date = '2023-11-01'
start_date = '1971-09-01' 
end_date = '2018-12-01' # As in Bianchi
# end_date = '2018-12-01' # Extended

fwd_df = fwd_df[(fwd_df['Date'] >= start_date) & (fwd_df['Date'] <= end_date)]
xr_df = xr_df[(xr_df['Date'] >= start_date) & (xr_df['Date'] <= end_date)]
macro_df = macro_df[(macro_df['Date'] >= start_date) & (macro_df['Date'] <= end_date)]

## Prepare variables
Y = xr_df.drop(columns = 'Date').values
X_fwd = fwd_df.drop(columns = 'Date').values
X_macro = macro_df.drop(columns = 'Date').values

# Scale variables
X_fwd_scaler = MinMaxScaler(feature_range=(-1,1))
X_macro_scaler = MinMaxScaler(feature_range=(-1,1))

X_fwd_scaled = X_fwd_scaler.fit_transform(X_fwd)
X_macro_scaled = X_macro_scaler.fit_transform(X_macro)

# Determine in-sample / out-of-sample split
oos_start_index = int(len(Y) * 0.85)
T = int(Y.shape[0])

## Set up and fit NN(1 layer, 3 neurons) model

def NN(X_f, X_m, Y, no, l1l2, dropout_rate, n_epochs=100):
    X_f_train, X_m_train, Y_train = X_f[:-1,:], X_m[:-1,:], Y[:-1,:] 
    X_f_test, X_m_test = X_f[-1,:].reshape(1,-1), X_m[-1,:].reshape(1,-1)
    Y_train = np.expand_dims(Y_train, axis=1)

    # Set seeds for replication
    tf.random.set_seed(777)
    np.random.seed(777)

    # Set up model architecture (1 hidden layer, 3 neurons)
    f_input = Input(shape=(X_f_train.shape[1],))
    m_input = Input(shape=(X_m_train.shape[1],))

    hidden_layer_1 = Dense(32, activation='relu', kernel_regularizer=l1_l2(l1l2))(m_input)
    drop_1 = Dropout(dropout_rate)(hidden_layer_1)
    hidden_layer_2 = Dense(16, activation='relu', kernel_regularizer=l1_l2(l1l2))(drop_1)
    drop_2 = Dropout(dropout_rate)(hidden_layer_2)
    hidden_layer_3 = Dense(8, activation='relu', kernel_regularizer=l1_l2(l1l2))(drop_2)
    drop_3 = Dropout(dropout_rate)(hidden_layer_3)

    add_f = Concatenate()([drop_3, f_input])

    output_layer = Dense(10, activation='linear')(add_f) 
    
    model = Model(inputs=[m_input, f_input], outputs=output_layer)

    # Compile model
    sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    dumploc = '/Users/avril/Desktop/Seminar/Python Code/dumploc_NN3_32_16_8'
    mcp = ModelCheckpoint(dumploc+'/BestModel_'+str(no)+'.keras',
                              monitor='val_loss',save_best_only=False)
    model.compile(optimizer=sgd_optimizer, loss='mse')

    model_hist = model.fit([X_m_train, X_f_train], Y_train, epochs=n_epochs, callbacks=[mcp], 
                           shuffle=True, validation_split = 0.15, verbose=0, )
    
    model = load_model(dumploc+'/BestModel_'+str(no)+'.keras')
    model.save(dumploc+'/BestModel_'+str(no)+'.keras')
    
    Y_pred = model.predict([X_m_test, X_f_test])
    val_loss = min(model_hist.history['val_loss'])

    return Y_pred, val_loss

# Generate predictions

param_grid = {
    'l1l2': [0.01, 0.001],
    'dropout_rate': [0.1, 0.3, 0.5]
}

re_estimation_freq = 3 # Re-estimation frequency for NN, in months
oos_iteration_indeces = range(oos_start_index, T, re_estimation_freq)
total_iterations = len(ParameterGrid(param_grid)) * (len(oos_iteration_indeces))
iteration_count = 0

all_Y_pred = []

for i in oos_iteration_indeces:
    best_val_loss = float('inf')
    best_params = None
    best_Y_pred = None

    for params in ParameterGrid(param_grid):
        
        iteration_count += 1
        print(f"Running iteration {iteration_count}/{total_iterations} | Testing params: {params}") 
        Y_pred, val_loss = NN(X_f=X_fwd_scaled[:i,:], X_m=X_macro_scaled[:i,:], Y = Y[:i,:], no=i, **params)  

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
            best_Y_pred = Y_pred
    
    all_Y_pred.append(best_Y_pred)
    
all_Y_pred = np.vstack(all_Y_pred) 

# Analyze model performance
all_R2_oos = {}
Y_test = Y[oos_start_index::re_estimation_freq,:]

for maturity in range(Y.shape[1]):
    r2_oos = ModelComparison_Rolling.R2OOS(y_true=Y_test[:, maturity], y_forecast=all_Y_pred[:, maturity])
    all_R2_oos[f"Maturity {maturity + 1}"] = r2_oos
    print(f"OOS R^2 Score for Maturity {maturity + 1}: {r2_oos:.4f}")