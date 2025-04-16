import pandas as pd

def compute_benchmark_prediction(xr_insample, xr_oos):
    benchmark_preds = []

    for i in range(len(xr_oos)):
        combined = pd.concat([xr_insample, xr_oos.iloc[:i+1]])
        #avg_val = combined.mean()  # Compute the mean of the combined data

        # This computes column-wise means from 12 months before
        avg_val = combined.iloc[:-12].mean() if len(combined) > 12 else combined.mean() 
        benchmark_preds.append(avg_val)

    # Convert list of Series (one per iteration) to a single DataFrames
    return pd.DataFrame(benchmark_preds, index=xr_oos.index)


if __name__ == "__main__":
    # Example usage
    xr = pd.read_excel("data-folder/!Data for forecasting/xr.xlsx")
    date_split = "1990-01-01"

    # Split insample and out-of-sample data
    xr_insample = xr[xr["Date"] < date_split].set_index("Date")
    xr_oos = xr[xr["Date"] >= date_split].set_index("Date")

    benchmark_predictions = compute_benchmark_prediction(xr_insample, xr_oos)
    # Save to excel 
    benchmark_predictions.to_excel("Extension code/Forecasting models/Saved preds/benchmark.xlsx", index=True)
    #print(benchmark_predictions.head())