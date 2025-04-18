import math
import pandas as pd
import matplotlib.pyplot as plt

def plot_oos_results(actual, predictions, benchmark, dates):
    num_models = len(predictions)
    
    # Use a single row for up to 3 plots; otherwise use a grid with 2 columns
    if num_models <= 3:
        fig, axs = plt.subplots(1, num_models, figsize=(5*num_models, 5))
        # When there's only one plot, axs is not a list, so we wrap it
        if num_models == 1:
            axs = [axs]
    else:
        ncols = 2
        nrows = math.ceil(num_models / ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(10, 5*nrows))
        axs = axs.flatten()
    
    for i, (model_name, pred_series) in enumerate(predictions.items()):
        axs[i].plot(dates, actual.values, label="Actual")
        axs[i].plot(dates, pred_series.values, linestyle='-.', label=f"{model_name}")
        axs[i].plot(dates, benchmark.values, label="Benchmark")
        axs[i].set_title(f"{model_name} forecasts")
        axs[i].set_xlabel("Date")
        axs[i].set_ylabel("Return Values")
        axs[i].legend()
        axs[i].grid(True)
    
    # Remove any unused subplots when using a grid layout
    if num_models > 3:
        for j in range(i+1, len(axs)):
            fig.delaxes(axs[j])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load the data
    
    '''
    reg_macro_preds = pd.read_excel("Extension code/Forecasting models/Saved preds/Regression preds/Macro_reg.xlsx")
    en_macro_preds = pd.read_excel("Extension code/Forecasting models/Saved preds/ElasticNet preds/Macro_en.xlsx")
    rf_macro_preds = pd.read_excel("Extension code/Forecasting models/Saved preds/RandomForest preds/Macro_rf.xlsx")
    #nn_macro_preds = pd.read_excel("Extension code/Forecasting models/Saved preds/NN preds/FWD_nn.xlsx")
    
    # Align the realized DataFrame date column with benchmark predictions
    benchmark_preds = pd.read_excel("Extension code/Forecasting models/Saved preds/benchmark.xlsx")
    realized = pd.read_excel("Extension code/Forecasting models/Saved preds/realized_xr.xlsx")
    realized.insert(0, "Date", benchmark_preds["Date"])
    reg_macro_preds.insert(0, "Date", benchmark_preds["Date"])
    en_macro_preds.insert(0, "Date", benchmark_preds["Date"])
    rf_macro_preds.insert(0, "Date", benchmark_preds["Date"])

    # Define the out-of-sample period
    start_oos = pd.to_datetime("1991-01-01")
    end_oos = pd.to_datetime("2023-11-01")

    # Align by date
    realized = realized[(realized["Date"] >= start_oos) & (realized["Date"] <= end_oos)]
    benchmark_preds = benchmark_preds[(benchmark_preds["Date"] >= start_oos) & (benchmark_preds["Date"] <= end_oos)]
    reg_macro_preds = reg_macro_preds[(reg_macro_preds["Date"] >= start_oos) & (reg_macro_preds["Date"] <= end_oos)]
    en_macro_preds = en_macro_preds[(en_macro_preds["Date"] >= start_oos) & (en_macro_preds["Date"] <= end_oos)]
    rf_macro_preds = rf_macro_preds[(rf_macro_preds["Date"] >= start_oos) & (rf_macro_preds["Date"] <= end_oos)]
    #nn_macro_preds = nn_macro_preds[(nn_macro_preds["Date"] >= start_oos) & (nn_macro_preds["Date"] <= end_oos)]
    dates = realized["Date"]


    # Pred dictionary
    pred_dictionary = {
        "Regression": reg_macro_preds["2 y"],
        "ElasticNet": en_macro_preds["2 y"],
        "RandomForest": rf_macro_preds["2 y"],
        #"NN": nn_macro_preds["2 y"]
    }
    

   
    plot_oos_results(
        actual=realized["2 y"],
        predictions=pred_dictionary,
        benchmark=benchmark_preds["2 y"],
        dates = dates
    )
    '''

    reg_macro_preds = pd.read_excel("Extension code/Forecasting models/Saved preds/Regression preds/Macro_reg.xlsx")
    reg_diff_macro_preds = pd.read_excel("Extension code/Forecasting models/Saved preds/Regression preds/diff_Macro.xlsx")
    benchmark_preds = pd.read_excel("Extension code/Forecasting models/Saved preds/benchmark.xlsx")
    realized = pd.read_excel("Extension code/Forecasting models/Saved preds/realized_xr.xlsx")
    dates = pd.read_excel("Extension code/Forecasting models/Saved preds/dates.xlsx")


    pred_dictionary = {
        "PCR (differenced)": reg_diff_macro_preds["2 y"],
        "PCR": reg_macro_preds["2 y"],
        #"RandomForest": rf_macro_preds["2 y"],
        #"NN": nn_macro_preds["2 y"]
    }

    plot_oos_results(
        actual=realized["2 y"],
        predictions=pred_dictionary,
        benchmark=benchmark_preds["2 y"],
        dates = dates
    )
