import pandas as pd
import numpy as np
import Data_preprocessing as data_prep
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import Roos
import compute_benchmark 
import ModelComparison_Rolling

## Upload and allign data

# Import yield and macro data, set in dataframe format with 'Date' column 
yields_df = pd.read_excel('/Users/avril/Desktop/Seminar/Data/Aligned_Yields_Extracted.xlsx')
forward_rates, xr = data_prep.process_yield_data(yields_df)
fwd_df, xr_df = pd.DataFrame(forward_rates), pd.DataFrame(xr)

macro_df = pd.read_csv('/Users/avril/Desktop/Seminar/Data/FRED-MD monthly 2024-12.csv')
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

## Set up and fit NN(1 layer, 3 neurons) model

def NN1_3(X, Y, n_epochs=10, no=1):
    X_train, Y_train = X[:-1,:], Y[:-1,:] 
    X_test, Y_test = X[-1,:].reshape(1,-1), Y[-1,:].reshape(1,-1)

    X_scaler = MinMaxScaler(feature_range=(-1,1))   
    X_scaled_train = X_scaler.fit_transform(X_train)
    
    # Keras requires 3D tuples for training.
    X_scaled_train = np.expand_dims(X_scaled_train, axis=1)
    Y_train = np.expand_dims(Y_train, axis=1)

    # Set seeds for replication
    tf.random.set_seed(777)
    np.random.seed(777)

    # Set up model architecture (1 hidden layer, 3 neurons)
    input_layer = Input(shape=(X_train.shape[1],))
    hidden_layer = Dense(3, activation='relu')(input_layer)
    output_layer = Dense(Y_train.shape[2], activation='linear')(hidden_layer) 
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile model
    sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9, decay=0.01, nesterov=True)
    dumploc = '/Users/avril/Desktop/Seminar/Python Code/dumploc'
    mcp = ModelCheckpoint(dumploc+'/BestModel_'+str(no)+'.keras',
                              monitor='val_loss',save_best_only=False)
    model.compile(optimizer=sgd_optimizer, loss='mse')

    model_hist = model.fit(X_train, Y_train, epochs=n_epochs, callbacks=[mcp], 
                           shuffle=True, validation_split = 0.15, verbose=0, )
    
    model = load_model(dumploc+'/BestModel_'+str(no)+'.keras')
    model.save(dumploc+'/BestModel_'+str(no)+'.hdf5')

    X_scaled_test = X_scaler.fit_transform(X_test)

    Y_pred = model.predict(X_scaled_test)

    return Y_pred, Y_test

all_Y_pred = []
all_Y_test = []

for i in range(oos_start_index, T):

    Y_pred, Y_test = NN1_3(X = X[:i,:], Y = Y[:i,:], no = 1)
    all_Y_pred.append(Y_pred)
    all_Y_test.append(Y_test)


all_Y_pred = np.vstack(all_Y_pred) 
all_Y_test = np.vstack(all_Y_test)  
all_R2_oos = {}

print("Shape of all_Y_pred:", all_Y_pred.shape)
print("Shape of all_Y_test:", all_Y_test.shape)
print("Shape of Y:", Y.shape)  # To compare

print("Package calculation")
for maturity in range(Y.shape[1]):
    r2_oos = ModelComparison_Rolling.R2OOS(y_true=all_Y_test[:, maturity], y_forecast=all_Y_pred[:, maturity])
    all_R2_oos[f"Maturity {maturity + 1}"] = r2_oos
    print(f"OOS R^2 Score for Maturity {maturity + 1}: {r2_oos:.4f}")