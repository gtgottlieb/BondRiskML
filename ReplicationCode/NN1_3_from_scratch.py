import numpy as np
import pandas as pd
import Data_preprocessing as data_prep
import ModelComparison_Rolling as comp_rolling
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate
from keras.optimizers import SGD
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

# Importing, generating and alligning the data

yields_df = pd.read_excel('/Users/avril/Desktop/Seminar/Data/Aligned_Yields_Extracted.xlsx')
forward_rates, xr = data_prep.process_yield_data(yields_df)
fwd_df, xr_df = pd.DataFrame(forward_rates), pd.DataFrame(xr)

start_date = xr_df['Date'].min()
end_date = "2023-11-01"

fwd_df = fwd_df[(fwd_df['Date'] <= end_date)]
xr_df = xr_df[(xr_df['Date'] <= end_date)]


macro_df = pd.read_csv('/Users/avril/Desktop/Seminar/Data/FRED-MD monthly 2024-12.csv')
macro_df = macro_df.drop(index=0) # Dropping the "Transform" row
macro_df = macro_df.rename(columns={'sasdate':'Date'})
macro_df['Date'] = pd.to_datetime(macro_df['Date'])
macro_df = macro_df[(macro_df['Date'] >= start_date) & (macro_df['Date'] <= end_date)]


# Applying NN (1 layer, 3 nodes)

X = fwd_df.drop(columns = 'Date').values
X_exog = macro_df.drop(columns = 'Date').values
Y = xr_df.drop(columns = 'Date').values

X_scaler = MinMaxScaler(feature_range=(-1,1))
X_scaled = X_scaler.fit_transform(X)

split_index = int(len(Y) * 0.85)
r2_scores = {}

for maturity in range(X.shape[1]):
    
    X_maturity = X_scaled[:, maturity].reshape(-1, 1)
    Y_maturity = Y[:, maturity].reshape(-1, 1)

    X_train, X_test = X_maturity[:split_index], X_maturity[split_index:]
    Y_train, Y_test = Y_maturity[:split_index], Y_maturity[split_index:]

    input_layer = Input(shape=(1,), name=f"{maturity + 1}_year")
    hidden_layer = Dense(3, activation='relu')(input_layer)
    output_layer = Dense(1, activation='linear')(hidden_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    sgd_optimizer = SGD(learning_rate=0.015, momentum=0.9, decay=0.01, nesterov=True)
    model.compile(optimizer=sgd_optimizer, loss='mse')

    model.fit(X_train, Y_train, epochs=100, verbose=0)
    Y_pred = model.predict(X_test)

    nan_indices = np.where(np.isnan(Y_pred))[0]
    if len(nan_indices) > 0:
        Y_test = np.delete(Y_test, nan_indices, axis=0)
        Y_pred = np.delete(Y_pred, nan_indices, axis=0)

    # Compute R² score
    r2 = r2_score(Y_test, Y_pred)
    r2_scores[f"Maturity {maturity + 1}"] = r2
    r2_OOS = comp_rolling.R2OOS(y_true = Y_test, y_forecast = Y_pred)

    print(f"R² Score for maturity {maturity + 1}: {r2_OOS:.4f}")

print("\nFinal R² Scores by Maturity:")
for maturity, score in r2_scores.items():
    print(f"{maturity}: {score:.4f}")