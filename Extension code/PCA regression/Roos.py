import numpy as np

def compute_oos_r2(rets, preds):
    """
    Compute the out-of-sample R² (oos R²) using a shifted expanding mean benchmark.

    Parameters:
        rets (np.ndarray): Actual returns.
        preds (np.ndarray): Predicted returns.

    Returns:
        float: The computed oos R² value.
    """
    # Shifted expanding mean benchmark
    benchmark = np.empty_like(rets)
    benchmark[0] = np.nan  # No history for first value

    for i in range(1, len(rets)):
        benchmark[i] = np.mean(rets[:i])

    # Mask out NaNs in both benchmark and predictions
    valid = ~np.isnan(benchmark) & ~np.isnan(preds)

    if np.sum(valid) < 10:
        print("Warning: too few valid data points to compute reliable R².")
        return np.nan

    ss_res = np.nansum((rets[valid] - preds[valid])**2)
    ss_tot = np.nansum((rets[valid] - benchmark[valid])**2)
    return 1 - ss_res / ss_tot