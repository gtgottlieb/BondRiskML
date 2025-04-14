import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

def compute_mse(predictions_dict, realized, dates):
    dates = dates.squeeze()
    model_mse_dict = {}
    for model_name, predictions in predictions_dict.items():
        mse_df = pd.DataFrame({'Date': dates})
        for col in predictions.columns:
            mse_values = []
            for i in range(len(predictions)):
                # Calculate RMSE for each row
                mse_value = mse(realized[col].iloc[:i+1], predictions[col].iloc[:i+1]) ** (1 / 2)
                mse_values.append(mse_value)
            mse_df[col] = mse_values
        mse_df['Average_MSE'] = mse_df.iloc[:, 1:].mean(axis=1)
        model_mse_dict[model_name] = mse_df
    return model_mse_dict

def plotting_mse(mse_dict_yield, mse_dict_macro):
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(14, 6))

    # Plot Yield models on the left
    ax = axes[0]
    for model_name, mse_df in mse_dict_yield.items():
        ax.plot(mse_df['Date'], mse_df['Average_MSE'], label=f'{model_name}')
    ax.set_title('RMSE averaged across predictions (Yield models)')
    ax.set_xlabel('Time')
    ax.set_ylabel('RMSE')
    ax.grid(True)
    ax.legend()

    # Plot Macro models on the right
    ax = axes[1]
    for model_name, mse_df in mse_dict_macro.items():
        ax.plot(mse_df['Date'], mse_df['Average_MSE'], label=f'{model_name}')
    ax.set_title('RMSE averaged across predictions (Yield+Macro models)')
    ax.set_xlabel('Time')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # Load FWD predictions
    fwd_preds = pd.read_excel("Extension code/Forecasting models/Saved preds/Regression preds/FWD_reg.xlsx")
    fwd_preds_en = pd.read_excel("Extension code/Forecasting models/Saved preds/ElasticNet preds/FWD_en.xlsx")
    fwd_preds_rf = pd.read_excel("Extension code/Forecasting models/Saved preds/RandomForest preds/FWD_rf.xlsx")

    # Load the benchmark
    benchmark = pd.read_excel("Extension code/Forecasting models/Saved preds/Benchmark.xlsx")
    # Subset benchmark to have the same columns as fwd_preds
    benchmark = benchmark[list(fwd_preds.columns)]

    # Create dictionary for FWD models
    fwd_dict = {
        'Benchmark': benchmark,
        'Regression': fwd_preds, 
        'ElasticNet': fwd_preds_en, 
        'RandomForest': fwd_preds_rf
    }

    # Load Macro predictions if needed
    macro_preds = pd.read_excel("Extension code/Forecasting models/Saved preds/Regression preds/Macro_reg.xlsx")
    macro_preds_en = pd.read_excel("Extension code/Forecasting models/Saved preds/ElasticNet preds/Macro_en.xlsx")
    macro_preds_rf = pd.read_excel("Extension code/Forecasting models/Saved preds/RandomForest preds/Macro_rf.xlsx")

    # Create dictionary for Macro models
    macro_dict = {
        'Benchmark': benchmark,
        'Regression': macro_preds, 
        'ElasticNet': macro_preds_en, 
        'RandomForest': macro_preds_rf
    }

    # Realized values and dates
    realized = pd.read_excel("Extension code/Forecasting models/Saved preds/realized_xr.xlsx")
    dates = pd.read_excel("Extension code/Forecasting models/Saved preds/dates.xlsx")

    # Run the MSE computation for FWD models (or change to macro_dict when needed)
    mse_fwd_dict = compute_mse(fwd_dict, realized, dates)
    mse_macro_dict = compute_mse(macro_dict, realized, dates)

    # Plot MSE values for FWD models
    plotting_mse(mse_fwd_dict, mse_macro_dict)


