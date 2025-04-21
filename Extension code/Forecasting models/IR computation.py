import numpy as np
import pandas as pd
<<<<<<< HEAD
from bayesian_shrinkage import bayesian_shrinkage
=======
>>>>>>> Gabriel's-Branch

def compute_z_scores_dynamic(predictions):
    """
    Compute z-scores dynamically at each time T using historical data up to T.

    Parameters:
    - predictions: A NumPy array of forecasted values (x_t(M)).

    Returns:
    - z_scores: A NumPy array of z-scores for each time step.
    """
    z_scores = []
    for t in range(len(predictions)):
        if t == 0:
            # No historical data for the first time step, z-score is undefined or 0
            z_scores.append(0)
        else:
            # Historical data up to time T
            historical_data = predictions[:t]
            historical_mean = np.mean(historical_data)
            historical_std = np.std(historical_data)
            
            # Avoid division by zero
            if historical_std == 0:
                z_scores.append(0)
            else:
                z_score = (predictions[t] - historical_mean) / historical_std
                z_scores.append(z_score)
    
    return np.array(z_scores)
    



def compute_information_ratio(z_scores, excess_returns):
    """
    Compute Information Ratio given z-scores and realized excess returns.
    """
    active_returns = z_scores * excess_returns  # Compute active strategy returns
    mean_return = np.mean(active_returns)
<<<<<<< HEAD
    std_return = np.std(active_returns)  # Sample standard deviation
=======
    std_return = np.std(active_returns)
>>>>>>> Gabriel's-Branch
    
    return mean_return / std_return if std_return != 0 else np.nan  # Avoid division by zero

if __name__ == "__main__":
    # Load predictions and realized excess returns
<<<<<<< HEAD
    realized = pd.read_excel("Extension code/Forecasting models/Saved preds/realized_xr.xlsx")
    bench = pd.read_excel("Extension code/Forecasting models/Saved preds/benchmark.xlsx")
    preds = pd.read_excel("Extension code/Forecasting models/Saved preds/Regression preds/diff_Macro.xlsx")
    bench = bench[preds.columns]
    #bayesian_preds = bayesian_shrinkage(bench, preds, prior_weight=0.5)
=======
    preds = pd.read_excel("data-folder/pca_bayes.xlsx").iloc[:, 1:]
    realized = pd.read_excel("data-folder/realized_xr.xlsx").iloc[:, 1:]
    
    print(preds.head())
    print(realized.head())
>>>>>>> Gabriel's-Branch


    z_scores = pd.DataFrame()
    info_ratios = {}

    for col in preds.columns:
        z_scores[col] = compute_z_scores_dynamic(preds[col].values)
        info_ratios[col] = compute_information_ratio(z_scores[col].values, realized[col].values)

    # Convert to DataFrame and display
    info_ratios_df = pd.DataFrame.from_dict(info_ratios, orient="index", columns=["IR"])
    print(info_ratios_df)
