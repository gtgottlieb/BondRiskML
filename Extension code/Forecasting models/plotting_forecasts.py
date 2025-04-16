import pandas as pd
import matplotlib.pyplot as plt

def plot_oos_results(actual, predictions, benchmark, dates):
    # Extract the oos dates from the xr.xlsx file.
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()

    for i, (model_name, pred_series) in enumerate(predictions.items()):
        axs[i].plot(dates, actual.values, label="Actual")
        axs[i].plot(dates, pred_series.values, linestyle='-.', label=f"{model_name} Predictions")
        axs[i].plot(dates, benchmark.values, label="Benchmark")
        axs[i].set_title(f"{model_name} forecasts")
        axs[i].set_xlabel("Date")
        axs[i].set_ylabel("Return Values")
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()

    '''
    # Loop over the predictions dictionary and plot each model's predictions
    plt.figure(figsize=(10, 6))
    plt.plot(dates, actual.values, linestyle="--", label="Actual")

    for model_name, pred_series in predictions.items():
        plt.plot(dates, pred_series.values, linestyle='-.', label=f"{model_name} Predictions")
    
    plt.plot(dates, benchmark.values, label="Benchmark")
    plt.title("Out-of-Sample Comparison")
    plt.xlabel("Date")
    plt.ylabel("Return Values")
    plt.legend()
    plt.grid(True)
    plt.show()
    '''

if __name__ == "__main__":
    # Load the data

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



