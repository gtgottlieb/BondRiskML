import numpy as np

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
