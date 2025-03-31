import numpy as np
import pandas as pd
import NNFuncBib as NFB
import Data_preprocessing as data_prep

yields_df = pd.read_excel('/Users/avril/Desktop/Seminar/Data/Aligned_Yields_Extracted.xlsx')
forward_rates, xr = data_prep.process_yield_data(yields_df)

xr['Date'] = pd.to_datetime(xr['Date'])
forward_rates['Date'] = pd.to_datetime(forward_rates['Date'])

start_date = xr['Date'].min()
end_date = "2023-11-01"

forward_rates = forward_rates[(forward_rates['Date'] <= end_date)]
xr = forward_rates[(xr['Date'] <= end_date)]

macro_df = pd.read_csv('/Users/avril/Desktop/Seminar/Data/FRED-MD monthly 2024-12.csv')
macro_df = macro_df.drop(index=0) # Dropping the "Transform" row
macro_df = macro_df.rename(columns={'sasdate':'Date'})
macro_df['Date'] = pd.to_datetime(macro_df['Date'])
macro_df = macro_df[(macro_df['Date'] >= start_date) & (macro_df['Date'] <= end_date)]

Ypred, val_loss = NFB.NN1LayerEnsemExog(X = macro_df, Xexog = forward_rates, Y = xr, no = 3, dumploc = '/Users/avril/Desktop/Seminar/Python Code/dumploc')
print('Y pred', Ypred)
print('val_loss', val_loss)