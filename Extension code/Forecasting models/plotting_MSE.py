import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

def compute_mse(predictions, realized, dates):
    # Use a list for better performance
    dates = dates.squeeze()
    mse_df = pd.DataFrame({'Date': dates})

    for col in predictions.columns:
        mse_values = []
        for i in range(len(predictions)):
            # Calculate MSE for each row
            mse_value = mse(realized[col].iloc[:i+1], predictions[col].iloc[:i+1]) #**(1 / 2) RMSE
            mse_values.append(mse_value)
        mse_df[col] = mse_values
    
    mse_df['Average_MSE'] = mse_df.iloc[:, 1:].mean(axis=1)

    return mse_df



if __name__ == "__main__":
    # Load predictions
    fwd_preds = pd.read_excel("Extension code/Forecasting models/Saved preds/Regression preds/FWD_reg.xlsx")
    macro_preds = pd.read_excel("Extension code/Forecasting models/Saved preds/Regression preds/Macro_reg.xlsx")
    realized = pd.read_excel("Extension code/Forecasting models/Saved preds/realized_xr.xlsx")
    dates = pd.read_excel("Extension code/Forecasting models/Saved preds/dates.xlsx")
    print(len(dates))

    # Run the MSE computation
    mse_fwd = compute_mse(fwd_preds, realized, dates)
    print(mse_fwd.tail())

    # Plot MSE values
    
    plt.figure(figsize=(10, 6))
    plt.plot(mse_fwd['Date'], mse_fwd['Average_MSE'], label='FWD Regression MSE')
    plt.title('Mean Squared Error (MSE) of FWD Regression Predictions')
    plt.xlabel('Time')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.show()
    


