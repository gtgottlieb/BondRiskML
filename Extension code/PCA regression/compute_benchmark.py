import pandas as pd

def compute_benchmark_prediction(xr_insample, xr_oos):
    benchmark_preds = []

    # Ensure xr_insample is a DataFrame

    for i in range(len(xr_oos)):
        combined = pd.concat([xr_insample, xr_oos.iloc[:i+1]]) 
        avg_val = combined.mean()  # This computes column-wise means
        benchmark_preds.append(avg_val)

    # Convert list of Series (one per iteration) to a single DataFrame
    return pd.DataFrame(benchmark_preds, index=xr_oos.index)